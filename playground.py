from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo 
import phi.api
import phi
from dotenv import load_dotenv
import os
from phi.playground import Playground,serve_playground_app
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")


websearch_agent = Agent(
    name="web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True
)

finanace_agent = Agent(
    name="Finance AI Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    instructions=["Use tables to display data"],
)

team_agent= Agent(
    team=[finanace_agent, websearch_agent],
    show_tool_calls=True,
    instructions=["Always include sources","Use tables to display data"],
    markdown=True,
    model=Groq(id="llama-3.1-70b-versatile")
)

#team_agent.print_response("summarize analyst recommendations for and share latest news for NVDA",stream=True)

app=Playground(agents=[team_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)
import streamlit as st
import zipfile
import tempfile
import os
from lxml import etree
import pandas as pd
from tableauhyperapi import HyperProcess, Telemetry, Connection, TableName
from io import BytesIO
import re
import time  # For progress simulation

# IMPROVEMENT: Added constants and helper functions at the top for better organization
DEFAULT_ROW_LIMIT = 50000
DEFAULT_TABLE_NAME = "Data"

class TableauParser:
    """Class to handle Tableau file parsing for better organization."""

    def __init__(self, twb_bytes, hyper_path=None):
        self.twb_bytes = twb_bytes
        self.hyper_path = hyper_path
        self.tree = None
        self._parse_twb()

    def _parse_twb(self):
        """Parse TWB bytes using lxml for consistency."""
        try:
            self.tree = etree.parse(BytesIO(self.twb_bytes))
        except Exception as e:
            raise ValueError(f"Failed to parse TWB file: {e}")

    def parse_worksheets_dashboards_fields(self):
        """Extract worksheets, dashboards, fields, and calculated fields."""
        root = self.tree.getroot()
        worksheets = [ws.get("name") for ws in root.findall(".//worksheet")]
        dashboards = [db.get("name") for db in root.findall(".//dashboard")]
        fields = []
        calculated_fields = []

        for col in root.findall(".//column"):
            name = col.get("name")
            if not name:
                continue
            if col.find("calculation") is not None:
                calculated_fields.append({
                    "name": col.get("name"),
                    "formula": col.find("calculation").get("formula")
                })
            else:
                fields.append({
                    "name": col.get("name"),
                    "role": col.get("role"),
                    "datatype": col.get("datatype")
                })
        return worksheets, dashboards, fields, calculated_fields

    def parse_calculated_fields(self):
        """Extract calculated fields."""
        root = self.tree.getroot()
        calculated_fields = []
        for col in root.findall(".//column"):
            calc = col.find("calculation")
            if calc is not None:
                calculated_fields.append({
                    "Name": col.get("caption") or col.get("name"),
                    "Formula": calc.get("formula"),
                    "Class": calc.get("class")
                })
        return calculated_fields

    def parse_parameters(self):
        """Extract parameters."""
        root = self.tree.getroot()
        parameters = []
        for col in root.findall(".//column"):
            if col.get("param-domain-type") is not None:
                parameters.append({
                    "Name": col.get("caption") or col.get("name"),
                    "Data Type": col.get("datatype"),
                    "Domain Type": col.get("param-domain-type"),
                    "Current Value": col.get("value")
                })
        return parameters

    def parse_filters(self):
        """Extract filters."""
        root = self.tree.getroot()
        filters = []
        for ws in root.findall(".//worksheet"):
            ws_name = ws.get("name")
            for flt in ws.findall(".//filter"):
                filter_info = {"Worksheet": ws_name, "Type": flt.get("class")}
                gf = flt.find("groupfilter")
                if gf is not None:
                    filter_info.update({"Field": gf.get("field"), "Members": gf.get("member")})
                rf = flt.find("rangefilter")
                if rf is not None:
                    filter_info.update({"Field": rf.get("field"), "Min": rf.get("min"), "Max": rf.get("max")})
                filters.append(filter_info)
        return filters

    def parse_measures(self):
        """Extract measures."""
        root = self.tree.getroot()
        measures = []
        for col in root.findall(".//column"):
            aggregation = col.get("aggregation")
            role = col.get("role")
            if aggregation or role == "measure":
                measures.append({
                    "Name": col.get("caption") or col.get("name"),
                    "Column": col.get("name"),
                    "Aggregation": aggregation
                })
        return measures

def read_hyper_data(hyper_path, limit=DEFAULT_ROW_LIMIT):
    """Read data from hyper file with error handling."""
    try:
        dataframes = []
        with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
            with Connection(hyper.endpoint, hyper_path) as connection:
                schemas = connection.catalog.get_schema_names()
                for schema in schemas:
                    tables = connection.catalog.get_table_names(schema)
                    for table in tables:
                        table_def = connection.catalog.get_table_definition(table)
                        columns = [col.name for col in table_def.columns]
                        query = f'SELECT * FROM {table} LIMIT {limit}'
                        rows = connection.execute_list_query(query)
                        df = pd.DataFrame(rows, columns=columns)
                        dataframes.append({"schema": table.schema_name, "table": table.name, "data": df})
        return dataframes
    except Exception as e:
        raise ValueError(f"Failed to read hyper data: {e}")

def detect_primary_keys(df):
    """Detect primary key candidates."""
    return [col for col in df.columns if df[col].isnull().sum() == 0 and df[col].nunique() == len(df)]

def classify_table(df):
    """Classify table as Fact or Dimension with refined heuristics."""
    numeric_cols = df.select_dtypes(include='number').shape[1]
    total_cols = df.shape[1]
    duplicate_pct = df.duplicated().sum() / len(df) if len(df) else 0
    # IMPROVEMENT: Added check for potential FKs (columns with low uniqueness)
    fk_like_cols = sum(1 for col in df.columns if df[col].nunique() / len(df) < 0.1)
    score = 0
    if total_cols > 0 and numeric_cols / total_cols > 0.5:
        score += 1
    if duplicate_pct > 0.1:
        score += 1
    if fk_like_cols > 0:  # Likely Dimension if FK-like columns exist
        score -= 1
    return "Fact" if score >= 1 else "Dimension"

def detect_relationships(tables):
    """Detect relationships with better overlap logic."""
    relationships = []
    for i, t1 in enumerate(tables):
        df1 = t1["data"]
        for t2 in tables[i + 1:]:
            df2 = t2["data"]
            common_cols = set(df1.columns).intersection(df2.columns)
            for col in common_cols:
                if df1[col].dtype == df2[col].dtype:
                    overlap = set(df1[col].dropna().unique()) & set(df2[col].dropna().unique())
                    overlap_pct = len(overlap) / min(df1[col].nunique(), df2[col].nunique()) if overlap else 0
                    if overlap_pct > 0.5: # IMPROVEMENT: Require >50% overlap for confidence
                        join_type = "1:1" if df1[col].nunique() == df2[col].nunique() else "1:Many"
                        relationships.append({
                            "From Table": t1["table"], "To Table": t2["table"], "Join Column": col, "Join Type": join_type
                        })
    return relationships

def extract_lod_expressions(calculated_fields):
    """Extract LOD expressions."""
    lods = []
    pattern = re.compile(r"\{.*?(fixed|include|exclude).*?:", re.IGNORECASE)
    for calc in calculated_fields:
        formula = calc["Formula"]
        if formula and pattern.search(formula):
            lods.append({"Name": calc["Name"], "LOD Expression": formula})
    return lods

def suggest_powerbi_visuals(worksheets, dashboards, fields, calculated_fields):
    """Suggest Power BI visuals based on Tableau elements."""
    suggestions = []
    for ws in worksheets:
        # Basic heuristic: If many measures, suggest a table/chart
        measures = [f for f in fields + calculated_fields if f.get("role") == "measure"]
        if len(measures) > 5:
            suggestions.append(f"For Worksheet '{ws}': Consider a Matrix or Table visual in Power BI.")
        else:
            suggestions.append(f"For Worksheet '{ws}': Use a Bar Chart or Line Chart.")
    for db in dashboards:
        suggestions.append(f"For Dashboard '{db}': Recreate as a Power BI Report Page with slicers for interactivity.")
    return suggestions

def is_simple_calc(formula: str) -> bool:
    """Check if calculation is simple."""
    complex_keywords = ["FIXED", "INCLUDE", "EXCLUDE", "WINDOW_", "RUNNING_", "RANK", "LOOKUP", "INDEX"]
    return not any(k.lower() in formula.lower() for k in complex_keywords)

def tableau_to_dax(formula: str, table_name=DEFAULT_TABLE_NAME):
    """Convert Tableau formula to DAX with improvements."""
    dax = formula
    # Aggregations
    dax = re.sub(r"SUM\s*\(([^)]+)\)", rf"SUM({table_name}[\1])", dax, flags=re.IGNORECASE)
    dax = re.sub(r"AVG\s*\(([^)]+)\)", rf"AVERAGE({table_name}[\1])", dax, flags=re.IGNORECASE)
    dax = re.sub(r"MIN\s*\(([^)]+)\)", rf"MIN({table_name}[\1])", dax, flags=re.IGNORECASE)
    dax = re.sub(r"MAX\s*\(([^)]+)\)", rf"MAX({table_name}[\1])", dax, flags=re.IGNORECASE)
    dax = re.sub(r"COUNT\s*\(([^)]+)\)", rf"COUNT({table_name}[\1])", dax, flags=re.IGNORECASE)
    dax = re.sub(r"COUNTD\s*\(([^)]+)\)", rf"DISTINCTCOUNT({table_name}[\1])", dax, flags=re.IGNORECASE)
    # IF THEN ELSE
    dax = re.sub(r"IF\s+(.+?)\s+THEN\s+(.+?)\s+ELSE\s+(.+?)\s+END", r"IF(\1, \2, \3)", dax, flags=re.IGNORECASE)
    # Field references (handle multi-table if table_name varies, but keep simple)
    dax = re.sub(r"\[([^]]+)\]", rf"[{table_name}][\1]", dax)
    # IMPROVEMENT: Basic LOD conversion (e.g., FIXED to SUMX with FILTER)
    dax = re.sub(r"\{FIXED\s*\[([^]]+)\]\s*:\s*SUM\s*\(([^)]+)\)\}", rf"SUMX(FILTER({table_name}, {table_name}[\1] = EARLIER({table_name}[\1])), {table_name}[\2])", dax, flags=re.IGNORECASE)
    return dax

def convert_measures_to_dax(measures, table_name=DEFAULT_TABLE_NAME):
    """Convert measures."""
    converted = []
    agg_map = {"sum": "SUM", "avg": "AVERAGE", "min": "MIN", "max": "MAX", "count": "COUNT", "countd": "DISTINCTCOUNT", "median": "MEDIAN"}
    for m in measures:
        agg = m["Aggregation"]
        if agg:
            dax_agg = agg_map.get(agg.lower())
            if dax_agg:
                dax = f"{m['Name']} = {dax_agg}({table_name}[{m['Column']}])"
                status = "Auto-converted"
            else:
                dax = None
                status = "Manual aggregation"
        else:
            dax = None
            status = "Implicit measure (manual review)"
        converted.append({"Measure Name": m["Name"], "Tableau Aggregation": agg, "DAX Measure": dax, "Status": status})
    return converted

def convert_calculated_fields(calculated_fields, table_name=DEFAULT_TABLE_NAME):
    """Convert calculated fields."""
    converted = []
    for calc in calculated_fields:
        formula = calc["Formula"]
        if not formula:
            continue
        if is_simple_calc(formula):
            dax_formula = tableau_to_dax(formula, table_name)
            status = "Auto-converted"
        else:
            dax_formula = None
            status = "Manual conversion required"
        converted.append({"Field Name": calc["Name"], "Tableau Formula": formula, "DAX Formula": dax_formula, "Status": status})
    return converted

def convert_filters_to_dax(filters, table_name=DEFAULT_TABLE_NAME):
    """Convert filters."""
    converted = []
    for f in filters:
        field = f.get("Field")
        filter_type = f.get("Type")
        members = f.get("Members")
        min_val = f.get("Min")
        max_val = f.get("Max")
        if filter_type == "categorical" and members:
            dax = f"FILTER({table_name}, {table_name}[{field}] IN {{{members}}})"
            status = "Auto-converted"
        elif filter_type == "range" and min_val and max_val:
            dax = f"FILTER({table_name}, {table_name}[{field}] >= {min_val} && {table_name}[{field}] <= {max_val})"
            status = "Auto-converted"
        else:
            dax = None
            status = "Manual review required"
        converted.append({"Field": field, "Type": filter_type, "DAX Filter": dax, "Status": status})
    return converted

def read_csv_excel_data(tmpdir, limit=DEFAULT_ROW_LIMIT):
    """Read data from CSV/Excel files in the extracted .twbx directory."""
    csv_excel_data = []
    try:
        for root_dir, _, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(('.csv', '.xlsx', '.xls')):
                    file_path = os.path.join(root_dir, file)
                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path, nrows=limit)
                    else:
                        df = pd.read_excel(file_path, nrows=limit)
                    csv_excel_data.append({
                        "file_name": file,
                        "data": df
                    })
    except Exception as e:
        st.warning(f"Error reading CSV/Excel data: {e}")
    return csv_excel_data

# Main App
st.set_page_config(page_title="Tableau to Power BI Analyzer", layout="wide")
st.title("Tableau to Power BI Analyzer")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Understanding", "Model Intelligence", "Tableau Logic", "DAX Conversion", "Power BI Recommendations"])

# IMPROVEMENT: Add a reset button
if st.sidebar.button("Reset Session"):
    st.session_state.clear()
    st.rerun()

if page == "Data Understanding":
    st.title("📊 Page 1: Tableau Data Understanding")
    uploaded_file = st.file_uploader("Upload Tableau Workbook (.twbx)", type=["twbx"])
    if uploaded_file:
        with st.spinner("Processing upload..."):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    twbx_path = os.path.join(tmpdir, uploaded_file.name)
                    with open(twbx_path, "wb") as f:
                        f.write(uploaded_file.read())
                    with zipfile.ZipFile(twbx_path, "r") as zip_ref:
                        zip_ref.extractall(tmpdir)
                    twb_path = None
                    hyper_path = None
                    for root_dir, _, files in os.walk(tmpdir):
                        for file in files:
                            if file.endswith(".twb"):
                                twb_path = os.path.join(root_dir, file)
                            if file.endswith(".hyper"):
                                hyper_path = os.path.join(root_dir, file)
                    
                    if not twb_path:
                        st.error("❌ Missing TWB file in the uploaded .twbx.")
                        st.stop()

                    # Read TWB as bytes
                    with open(twb_path, "rb") as f:
                        twb_bytes = f.read()
                    st.session_state["twb_bytes"] = twb_bytes

                    parser = TableauParser(twb_bytes, hyper_path)
                    worksheets, dashboards, fields, calculated_fields = parser.parse_worksheets_dashboards_fields()
                    
                    # Try reading Hyper data, fallback to CSV/Excel if Hyper fails or is missing
                    tables = []
                    if hyper_path:
                        try:
                            tables = read_hyper_data(hyper_path)
                        except Exception as e:
                            st.warning(f"Could not read Hyper file: {e}. Attempting to read CSV/Excel files.")
                            tables = read_csv_excel_data(tmpdir)
                    else:
                        st.info("No .hyper file found. Attempting to read CSV/Excel files.")
                        tables = read_csv_excel_data(tmpdir)
                    
                    if not tables:
                        st.warning("No data found in Hyper, CSV, or Excel files.")
                        
                    st.session_state["tables"] = tables

                    st.success("✅ Tableau file analyzed successfully")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Worksheets", len(worksheets))
                    col2.metric("Total Dashboards", len(dashboards))
                    col3.metric("Total Fields", len(fields))

                    st.markdown("## 📄 Worksheets")
                    st.write(worksheets)

                    st.markdown("## 🖥️ Dashboards")
                    st.write(dashboards)

                    st.markdown("## 📋 Fields")
                    st.dataframe(pd.DataFrame(fields))

                    st.markdown("## 🔢 Calculated Fields")
                    st.dataframe(pd.DataFrame(calculated_fields))

                    st.markdown("## 📊 Duplicate Column Names")
                    for t in tables:
                        df = t["data"]
                        total_rows = len(df)
                        duplicate_rows = df.duplicated().sum()
                        duplicate_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows else 0
                        st.markdown(f"### 🗂️ Table: {t['schema']}.{t['table']}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Rows", total_rows)
                        col2.metric("Duplicate Rows", duplicate_rows)
                        col3.metric("Duplicate %", f"{duplicate_pct}%")
                        st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

elif page == "Model Intelligence":
    st.title("🧠 Page 2: Model Intelligence")
    tables = st.session_state.get("tables")
    if not tables:
        st.warning("Please upload a Tableau file on Page 1 first.")
        st.stop()

    st.subheader("🔍 Table Classification & Keys")
    for t in tables:
        df = t["data"]
        table_type = classify_table(df)
        pk_candidates = detect_primary_keys(df)
        with st.expander(f"🗂️ {t['schema']}.{t['table']}"):
            col1, col2 = st.columns(2)
            col1.metric("Table Type", table_type)
            col2.metric("Rows", len(df))
            st.markdown(f"**Primary Key Candidates**")
            if pk_candidates:
                st.success(pk_candidates)
            else:
                st.warning("No strong primary key detected")
            st.markdown(f"**Column Data Types**")
            st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}), use_container_width=True)

    st.subheader("🔗 Suggested Relationships")
    relationships = detect_relationships(tables)
    if relationships:
        st.dataframe(pd.DataFrame(relationships), use_container_width=True)
    else:
        st.info("No relationships inferred automatically.")

elif page == "Tableau Logic":
    st.title("📐 Page 3: Tableau Logic Extraction")
    twb_bytes = st.session_state.get("twb_bytes")
    if not twb_bytes:
        st.error("TWB metadata not available. Please re-upload Tableau file.")
        st.stop()

    parser = TableauParser(twb_bytes)
    calcs = parser.parse_calculated_fields()
    params = parser.parse_parameters()
    filters = parser.parse_filters()
    measures = parser.parse_measures()
    lods = extract_lod_expressions(calcs)

    st.subheader("🔢 Calculated Fields")
    if calcs:
        st.dataframe(calcs, use_container_width=True)
    else:
        st.info("No calculated fields found.")

    st.subheader("🎛️ Parameters")
    if params:
        st.dataframe(params, use_container_width=True)
    else:
        st.info("No parameters found.")

    st.subheader("🔎 Filters")
    if filters:
        st.dataframe(filters, use_container_width=True)
    else:
        st.info("No filters found.")

    st.subheader("📈 Measures")
    if measures:
        st.dataframe(measures, use_container_width=True)
    else:
        st.info("No measures found.")

    st.subheader("📦 Level of Detail (LOD) Expressions")
    if lods:
        st.dataframe(lods, use_container_width=True)
        st.warning("⚠️ LOD expressions usually require manual DAX conversion.")
    else:
        st.success("✅ No LOD expressions detected – high auto-conversion success.")

elif page == "DAX Conversion":
    st.title("🔄 Step 4: Tableau -> DAX Conversion")

    twb_bytes = st.session_state.get("twb_bytes")
    if not twb_bytes:
        st.error("TWB metadata not available. Please re-upload Tableau file.")
        st.stop()

    parser = TableauParser(twb_bytes)
    calcs = parser.parse_calculated_fields()
    measures = parser.parse_measures()
    filters = parser.parse_filters()
    params = parser.parse_parameters()
    lods = extract_lod_expressions(calcs)

    # IMPROVEMENT: Allow user to select table name for DAX
    table_options = [str(t["table"]) for t in st.session_state.get("tables", [])] or [DEFAULT_TABLE_NAME] # Convert to string
    selected_table = st.selectbox("Select Table for DAX Conversion", table_options, index=0)

    converted_calcs = convert_calculated_fields(calcs, selected_table)
    converted_measures = convert_measures_to_dax(measures, selected_table)
    converted_filters = convert_filters_to_dax(filters, selected_table)

    st.subheader("📝 Calculated Field Conversion")
    st.dataframe(converted_calcs, use_container_width=True)
    auto_count = sum(1 for c in converted_calcs if c["Status"] == "Auto-converted")
    manual_count = len(converted_calcs) - auto_count
    st.success(f"✅ Auto-converted: {auto_count}")
    st.warning(f"⚠️ Manual review required: {manual_count}")

    st.subheader("📈 Measure Conversion")
    st.dataframe(converted_measures, use_container_width=True)
    auto = sum(1 for m in converted_measures if m["Status"] == "Auto-converted")
    manual = len(converted_measures) - auto
    st.success(f"✅ Measures auto-converted: {auto}")
    st.warning(f"⚠️ Measures needing review: {manual}")

    st.subheader("🔎 Filter Conversion")
    if converted_filters:
        st.dataframe(converted_filters, use_container_width=True)
        auto = sum(1 for f in converted_filters if f["Status"] == "Auto-converted")
        manual = len(converted_filters) - auto
        st.success(f"✅ Filters auto-converted: {auto}")
        st.warning(f"⚠️ Filters needing manual review: {manual}")
    else:
        st.info("No filters detected or all filters require manual review.")

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Existing sheets
        pd.DataFrame(converted_calcs).to_excel(writer, sheet_name="Calculated Fields", index=False)
        pd.DataFrame(converted_measures).to_excel(writer, sheet_name="Measures", index=False)
        pd.DataFrame(converted_filters).to_excel(writer, sheet_name="Filters", index=False)
        pd.DataFrame(params).to_excel(writer, sheet_name="Parameters", index=False)
        pd.DataFrame(lods).to_excel(writer, sheet_name="LOD Expressions", index=False)

        # New: Add data from Hyper tables
        tables = st.session_state.get("tables", [])
        for t in tables:
            sheet_name = f"Table_{str(t['table'])[:20]}" # Convert to string and truncate
            t["data"].to_excel(writer, sheet_name=sheet_name, index=False)

        # Summary sheet - Fixed to have equal-length lists
        num_tables = len(tables)
        total_conversions = len(converted_calcs) + len(converted_measures) + len(converted_filters)
        auto_converted = auto_count + auto + (sum(1 for f in converted_filters if f["Status"] == "Auto-converted") if converted_filters else 0)
        summary_data = {
            "Hyper Tables": [f"{str(t['schema'])}.{str(t['table'])}" for t in tables],
            "Total Conversions": [total_conversions] * num_tables,
            "Auto-Converted": [auto_converted] * num_tables # Repeat for each table
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

    output.seek(0)

    st.download_button(
        label="Download Excel Report",
        data=output,
        file_name="PowerBI_Inventory_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

elif page == "Power BI Recommendations":
    st.title("💡 Page 5: Power BI Recommendations")

    tables = st.session_state.get("tables", [])
    twb_bytes = st.session_state.get("twb_bytes")
    if not tables or not twb_bytes:
        st.warning("Please upload and process a Tableau file first.")
        st.stop()

    parser = TableauParser(twb_bytes)
    calcs = parser.parse_calculated_fields()
    measures = parser.parse_measures()
    relationships = detect_relationships(tables)

    st.subheader("💡 Migration Tips")
    st.markdown("""
    **Data Model:** Import tables as-is. Use inferred relationships to build your model.
    **Measures:** Create measures in Power BI using the converted DAX.
    **Visuals:** Recreate worksheets as Power BI reports. Use slicers for filters.
    **Parameters:** Convert to Power BI parameters for dynamic values.
    **LOD Expressions:** These are complex; consider using DAX functions like SUMX or CALCULATE.
    """)

    st.subheader("📊 Suggested Measures & Hierarchies")
    # IMPROVEMENT: Basic recommendations based on data
    for t in tables:
        df = t["data"]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            st.markdown(f"**For Table {t['table']}**: Consider creating measures like SUM of {numeric_cols[0]} or AVERAGE of {numeric_cols[0]}.")
        date_cols = df.select_dtypes(include="datetime").columns.tolist()
        if date_cols:
            st.markdown(f"**Hierarchy Suggestion:** Create a date hierarchy from {date_cols[0]} (Year, Quarter, Month).")

    st.subheader("🔗 Relationship Setup in Power BI")
    if relationships:
        st.dataframe(pd.DataFrame(relationships), use_container_width=True)
        st.info("In Power BI, go to Model view and create relationships based on these suggestions.")
    else:
        st.info("No relationships suggested; ensure manual setup if needed.")

    # Overall success rate
    total_conversions = len(calcs) + len(measures) + len(parser.parse_filters())
    auto_conversions = sum(1 for c in convert_calculated_fields(calcs) if c["Status"] == "Auto-converted") + \
                       sum(1 for m in convert_measures_to_dax(measures) if m["Status"] == "Auto-converted") + \
                       sum(1 for f in convert_filters_to_dax(parser.parse_filters()) if f["Status"] == "Auto-converted")
    success_rate = round((auto_conversions / total_conversions * 100), 2) if total_conversions else 0
    st.metric("Estimated Auto-Conversion Success Rate", f"{success_rate}%")
