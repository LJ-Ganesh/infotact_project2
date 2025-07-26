import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DAILY_CAPACITY_DEFAULT = 50 # Default capacity
MIN_DAILY_CAPACITY = 5 # New minimum workload
MAX_DAILY_CAPACITY = 100 # New maximum workload

# --- Load pre-trained models ---
try:
    category_model = joblib.load("best_category_model.pkl")
    priority_model = joblib.load("best_priority_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    label_encoder = joblib.load("category_encoder.pkl")
except FileNotFoundError:
    st.error("""
        Model files not found! Please ensure the following files are in the same directory as this script:
        - `best_category_model.pkl`
        - `best_priority_model.pkl`
        - `vectorizer.pkl`
        - `category_encoder.pkl`
        Please provide these files to run the application.
    """)
    st.stop() # Stop the app if models can't be loaded

# --- Session State Initialization ---
# Initialize all session state variables at the very beginning
if "tasks" not in st.session_state:
    st.session_state["tasks"] = []
if "daily_capacity" not in st.session_state:
    st.session_state["daily_capacity"] = DAILY_CAPACITY_DEFAULT
# Key for the text input widget itself, to control its value for clearing
if "task_input_widget_key" not in st.session_state:
    st.session_state["task_input_widget_key"] = ""
# For navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Schedule Planner"


# --- Page Config ---
st.set_page_config(page_title="Schedule Buddy", page_icon="🤖", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
    background-color: #f3e8ff;
}
.main {
    background-color: #f9f4ff;
    padding: 2rem;
    border-radius: 10px;
}
.title {
    color: #4a148c;
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 30px;
}
.section-title {
    color: #6a1b9a;
    font-size: 1.5rem;
    font-weight: bold;
    margin-top: 30px;
    margin-bottom: 15px;
}
.task-card {
    background: #ffffff;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 8px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.sidebar .sidebar-content {
    background-color: #e1bee7;
}
.stButton > button {
    background-color: #7b1fa2;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 1rem;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #4a0072;
}
.stTextInput > div > div > input {
    border-radius: 8px;
    border: 1px solid #7b1fa2;
    padding: 10px;
    font-size: 1rem;
}
.stMetric > div {
    background-color: #d1c4e9; /* Medium lavender */
    border-left: 5px solid #6a1b9a; /* Deep purple border */
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    color: #1a1a1a; /* Dark text for contrast */
    font-weight: bold;
}
.stMetric > div:hover {
    box-shadow: 0 4px 8px rgba(106, 27, 154, 0.3);
}
/* Style for completed tasks in the dataframe */
.stDataFrame td.completed {
    text-decoration: line-through;
    color: #888;
    background-color: #f1f8e9 !important; /* Lighter green for completed */
}
/* Styles for disabled button */
.stButton > button[aria-disabled="true"] {
    background-color: #d1c4e9;
    cursor: not-allowed;
    color: #666666;
}
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def priority_label(p):
    return {0: "Low", 1: "Medium", 2: "High"}.get(p, "Unknown")

def priority_numeric(p_label):
    return {"Low": 0, "Medium": 1, "High": 2}.get(p_label, 0) # Default to Low if unknown

def process_task_input_callback():
    """
    Callback function for the text input or add button.
    Processes the task and clears the input.
    """
    task_description = st.session_state["task_input_widget_key"]
    if task_description.strip():
        try:
            X_new = vectorizer.transform([task_description])
            cat_pred_numeric = category_model.predict(X_new)[0]
            prio_pred_numeric = priority_model.predict(X_new)[0]

            category_predicted = label_encoder.inverse_transform([cat_pred_numeric])[0]
            priority_predicted = priority_label(prio_pred_numeric)

            st.session_state["tasks"].append({
                "ID": max([t["ID"] for t in st.session_state["tasks"]]) + 1 if st.session_state["tasks"] else 1,
                "Task": task_description.strip(),
                "Category": category_predicted,
                "Priority": priority_predicted,
                "Priority_Numeric": prio_pred_numeric,
                "Word_Count": len(task_description.split()),
                "Completed": False # New field for completion status
            })
            st.success("Task added successfully! 🎉")
            st.session_state["task_input_widget_key"] = "" # Clear the input field
        except Exception as e:
            st.error(f"Error processing task: {e}")
            st.warning("Please ensure your model files are valid and the input is suitable.")
    # No else here, because the button's disabled state handles empty input

# Function to clear all tasks
def clear_all_tasks():
    st.session_state["tasks"] = []
    st.success("All tasks cleared! Starting fresh. ✨")


# Function to process and schedule tasks, always returns a DataFrame
def get_processed_tasks_df(tasks_list, daily_capacity):
    if not tasks_list:
        return pd.DataFrame() # Return empty DataFrame if no tasks

    df = pd.DataFrame(tasks_list)

    # Ensure these columns are always calculated for any use of the DataFrame
    df['Priority_Numeric'] = df['Priority'].apply(priority_numeric)
    df['Word_Count'] = df['Task'].apply(lambda x: len(str(x).split()))
    df['Workload_Score'] = (df['Priority_Numeric'] + 1) * df['Word_Count']

    # Store a unique identifier for each row *before* sorting for scheduling
    df['__original_df_index__'] = df.index

    df_for_scheduling = df.sort_values(by="Workload_Score", ascending=False).copy()

    assigned_days = []
    current_load = 0
    current_day = 1

    for idx, row in df_for_scheduling.iterrows():
        score = row["Workload_Score"]
        if current_load + score > daily_capacity and current_load > 0:
            current_day += 1
            current_load = 0
        assigned_days.append(current_day)
        current_load += score

    df_for_scheduling["Day_Assigned"] = assigned_days

    # Merge 'Day_Assigned' back to the original `df` based on '__original_df_index__'
    # Create a temporary DataFrame for merging with just the necessary columns
    merge_cols_df = df_for_scheduling[['__original_df_index__', 'Day_Assigned', 'Workload_Score']]

    # Perform the merge. Using pd.merge is safer than set_index/join for complex merges.
    # First, ensure original df doesn't have old Day_Assigned/Workload_Score if they were added previously
    df_temp = df.drop(columns=['Day_Assigned', 'Workload_Score'], errors='ignore')
    
    df_merged = pd.merge(df_temp, merge_cols_df, on='__original_df_index__', how='left')
    
    # Drop the temporary index column
    df_merged = df_merged.drop(columns=['__original_df_index__'])

    # Re-sort for consistent display
    df_merged = df_merged.sort_values(by=["Day_Assigned", "Workload_Score"], ascending=[True, False]).reset_index(drop=True)

    return df_merged


# --- Title ---
st.markdown("<div class='title'>Schedule Buddy 🤖</div>", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/5900/5900785.png", width=120)
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to:",
    ["📅 Schedule Planner", "📊 Dashboard Overview", "📈 Insights & Charts"],
    key="page_selector"
)
st.session_state["current_page"] = page_selection

# --- Settings Section in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='section-title'>⚙️ Settings</div>", unsafe_allow_html=True)
new_capacity = st.sidebar.slider(
    "Adjust Daily Workload Capacity",
    min_value=MIN_DAILY_CAPACITY, # Adjusted min range
    max_value=MAX_DAILY_CAPACITY, # Adjusted max range
    value=st.session_state["daily_capacity"],
    step=50,
    key="daily_capacity_slider"
)
if new_capacity != st.session_state["daily_capacity"]:
    st.session_state["daily_capacity"] = new_capacity
    st.sidebar.success(f"Daily capacity set to {new_capacity} units.")


# --- Main Content Area based on Navigation ---
if st.session_state["current_page"] == "📅 Schedule Planner":
    st.header("📅 Your Personal Schedule Planner")

    # --- Clear All Tasks Button (Moved to top of this section) ---
    if st.session_state["tasks"]: # Only show if there are tasks
        if st.button("Clear All Tasks 🗑️", key="clear_all_button_top", help="Removes all tasks from the list"):
            clear_all_tasks()

    # --- Input Area ---
    st.markdown("<div class='section-title'>🖋️ Enter a Task Description:</div>", unsafe_allow_html=True)

    st.text_input(
        "What's your task today?",
        key="task_input_widget_key",
        on_change=process_task_input_callback,
        value=st.session_state["task_input_widget_key"],
        placeholder="e.g., 'Prepare presentation slides for client meeting'"
    )

    task_input_is_empty = not st.session_state["task_input_widget_key"].strip()

    if st.button("Add Task ➕", key="add_task_button", disabled=task_input_is_empty, help="Click to add your task. Disabled if input is empty."):
        process_task_input_callback()


    # --- Display & Manage Tasks ---
    if st.session_state["tasks"]:
        df = get_processed_tasks_df(st.session_state["tasks"], st.session_state["daily_capacity"])

        st.markdown("<div class='section-title'>📋 Your Current Tasks</div>", unsafe_allow_html=True)

        edited_df = st.data_editor(
            df[['ID', 'Task', 'Category', 'Priority', 'Completed']],
            column_config={
                "ID": st.column_config.NumberColumn("ID", help="Unique Task ID", disabled=True),
                "Completed": st.column_config.CheckboxColumn(
                    "Done?",
                    help="Mark task as complete",
                    default=False,
                ),
                "Task": st.column_config.TextColumn("Task Description", help="The description of your task"),
                "Category": st.column_config.SelectboxColumn(
                    "Predicted Category",
                    options=label_encoder.classes_.tolist(),
                    required=True,
                    help="AI Predicted Category (editable)"
                ),
                "Priority": st.column_config.SelectboxColumn(
                    "Predicted Priority",
                    options=["Low", "Medium", "High"],
                    required=True,
                    help="AI Predicted Priority (editable)"
                ),
            },
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            key="task_data_editor"
        )

        needs_rerun_after_edit = False
        updated_tasks_from_editor = edited_df.to_dict('records')
        new_session_tasks = []
        original_task_map = {task['ID']: task for task in st.session_state["tasks"]}

        # Check for changes in existing tasks and new rows
        for edited_row in updated_tasks_from_editor:
            task_id = edited_row['ID']
            if task_id in original_task_map:
                original_task = original_task_map[task_id].copy()
                if (original_task['Task'] != edited_row['Task'] or
                    original_task['Category'] != edited_row['Category'] or
                    original_task['Priority'] != edited_row['Priority'] or
                    original_task['Completed'] != edited_row['Completed']):
                    original_task.update(edited_row)
                    needs_rerun_after_edit = True
                new_session_tasks.append(original_task)
            else:
                # This logic is for new rows added directly in the data_editor.
                # It assigns a new ID and attempts to predict.
                st.warning(f"New row detected in table: '{edited_row.get('Task', 'Unnamed Task')}'. "
                           "Attempting to process it. For full prediction, use the text input above.")
                new_id = max([t["ID"] for t in st.session_state["tasks"]]) + 1 if st.session_state["tasks"] else 1
                edited_row['ID'] = new_id

                if 'Task' in edited_row and edited_row['Task'].strip():
                    try:
                        temp_X_new = vectorizer.transform([edited_row['Task']])
                        temp_cat_pred_numeric = category_model.predict(temp_X_new)[0]
                        temp_prio_pred_numeric = priority_model.predict(temp_X_new)[0]
                        edited_row['Category'] = label_encoder.inverse_transform([temp_cat_pred_numeric])[0]
                        edited_row['Priority'] = priority_label(temp_prio_pred_numeric)
                    except Exception: # Catch any prediction errors
                        edited_row['Category'] = edited_row.get('Category', 'Uncategorized')
                        edited_row['Priority'] = edited_row.get('Priority', 'Low')
                else:
                    edited_row['Category'] = edited_row.get('Category', 'Uncategorized')
                    edited_row['Priority'] = edited_row.get('Priority', 'Low')
                
                # Ensure all required fields for task dict are present
                edited_row['Priority_Numeric'] = priority_numeric(edited_row['Priority'])
                edited_row['Word_Count'] = len(str(edited_row['Task']).split())
                edited_row['Workload_Score'] = (edited_row['Priority_Numeric'] + 1) * edited_row['Word_Count']
                edited_row['Completed'] = edited_row.get('Completed', False)

                new_session_tasks.append(edited_row)
                needs_rerun_after_edit = True

        # Check for deleted rows (by comparing IDs)
        current_ids_in_editor = {row['ID'] for row in updated_tasks_from_editor}
        original_ids_in_session = {task['ID'] for task in st.session_state["tasks"]}

        if len(current_ids_in_editor) != len(original_ids_in_session):
            needs_rerun_after_edit = True # A row was added or deleted directly in the editor

        st.session_state["tasks"] = new_session_tasks

        if needs_rerun_after_edit:
            st.info("Task table updated. Recalculating schedule...🔄")
            # A rerun will naturally be triggered because st.session_state["tasks"] changed.

        # Deletion functionality (Separate from data_editor for explicit control)
        st.markdown("<div class='section-title'>🗑️ Delete Tasks</div>", unsafe_allow_html=True)
        if not df.empty:
            task_id_map = {f"{task['Task']} (ID: {task['ID']})": task['ID'] for task in st.session_state["tasks"]}

            selected_display_names = st.multiselect(
                "Select tasks to delete:",
                options=list(task_id_map.keys()),
                key="delete_multiselect"
            )

            task_ids_to_delete = [task_id_map[display_name] for display_name in selected_display_names]

            if st.button("Delete Selected Tasks 🗑️", key="confirm_delete_button", disabled=not bool(task_ids_to_delete)):
                if task_ids_to_delete:
                    st.session_state["tasks"] = [
                        task for task in st.session_state["tasks"]
                        if task["ID"] not in task_ids_to_delete
                    ]
                    st.success(f"Successfully deleted {len(task_ids_to_delete)} task(s). ✅")
                # No else block needed as button is disabled if no tasks selected


        # --- Schedule Timetable ---
        st.markdown("<div class='section-title'>📅 Task Timetable</div>", unsafe_allow_html=True)
        def highlight_completed(row):
            style = ['' for _ in row.index]
            if row['Completed']:
                try:
                    task_col_idx = list(row.index).index('Task')
                    style[task_col_idx] = 'text-decoration: line-through; color: #666666;'
                except ValueError:
                    pass
                return [s + 'background-color: #e6ffe6;' for s in style]
            return style

        st.dataframe(
            df[["Task", "Category", "Priority", "Day_Assigned", "Completed"]].style.apply(highlight_completed, axis=1),
            use_container_width=True,
            hide_index=True
        )


        # --- Workload Summary Chart ---
        st.markdown("<div class='section-title'>📊 Workload Summary</div>", unsafe_allow_html=True)
        workload_per_day = df.groupby("Day_Assigned")["Workload_Score"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="Day_Assigned", y="Workload_Score", data=workload_per_day, palette="viridis", ax=ax)
        plt.axhline(st.session_state["daily_capacity"], color='red', linestyle='--', label=f'Daily Limit ({st.session_state["daily_capacity"]})')
        plt.xlabel("Day")
        plt.ylabel("Workload Score")
        plt.title("Daily Workload Distribution")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)


        # --- Output Summary ---
        st.markdown("<div class='section-title'>🌟 Key Metrics</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Tasks", value=len(df))
        with col2:
            st.metric(label="Total Days Scheduled", value=df['Day_Assigned'].max())
        with col3:
            st.metric(label="Average Workload/Day", value=f"{workload_per_day['Workload_Score'].mean():.2f}")


    else: # No tasks in session state for Schedule Planner
        st.info("No tasks added yet. Start by entering a task description above!")

elif st.session_state["current_page"] == "📊 Dashboard Overview":
    st.header("📊 Dashboard Overview")
    st.write("This page provides a high-level summary and key metrics of your tasks.")

    if st.session_state["tasks"]:
        df = get_processed_tasks_df(st.session_state["tasks"], st.session_state["daily_capacity"]) # Always process df here

        total_tasks = len(df)
        completed_tasks = df['Completed'].sum()
        pending_tasks = total_tasks - completed_tasks

        st.subheader("Task Completion Status")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Total Tasks", total_tasks, delta=None)
        with col_d2:
            st.metric("Completed Tasks", completed_tasks, delta=None)
        with col_d3:
            st.metric("Pending Tasks", pending_tasks, delta=None)

        st.subheader("Tasks by Category")
        category_counts = df['Category'].value_counts()
        if not category_counts.empty:
            fig_cat, ax_cat = plt.subplots(figsize=(8, 4))
            sns.barplot(x=category_counts.index, y=category_counts.values, palette="cubehelix", ax=ax_cat)
            ax_cat.set_xlabel("Category")
            ax_cat.set_ylabel("Number of Tasks")
            ax_cat.set_title("Distribution of Tasks by Category")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_cat)
        else:
            st.info("No categories to display yet.")

        st.subheader("Tasks by Priority")
        priority_counts = df['Priority'].value_counts().reindex(["High", "Medium", "Low"])
        priority_counts = priority_counts.dropna()
        if not priority_counts.empty:
            fig_prio, ax_prio = plt.subplots(figsize=(8, 4))
            ax_prio.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
            ax_prio.set_title("Distribution of Tasks by Priority")
            st.pyplot(fig_prio)
        else:
            st.info("No priorities to display yet.")

    else:
        st.info("Add some tasks on the 'Schedule Planner' page to see dashboard insights!")

elif st.session_state["current_page"] == "📈 Insights & Charts":
    st.header("📈 Deeper Insights & Charts")
    st.write("This page provides more detailed analytical charts about your task patterns and workload.")

    if st.session_state["tasks"]:
        df = get_processed_tasks_df(st.session_state["tasks"], st.session_state["daily_capacity"]) # Always process df here

        st.subheader("Workload Trend Over Days")
        daily_workload_trend = df.groupby('Day_Assigned')['Workload_Score'].sum().reset_index()
        if not daily_workload_trend.empty:
            fig_trend, ax_trend = plt.subplots(figsize=(8, 4))
            sns.lineplot(x='Day_Assigned', y='Workload_Score', data=daily_workload_trend, marker='o', ax=ax_trend)
            plt.axhline(st.session_state["daily_capacity"], color='red', linestyle='--', label='Daily Limit')
            ax_trend.set_xlabel("Day Assigned")
            ax_trend.set_ylabel("Total Workload Score")
            ax_trend.set_title("Workload Score Trend Across Scheduled Days")
            ax_trend.legend()
            st.pyplot(fig_trend)
        else:
            st.info("No daily workload trend to display yet.")


        st.subheader("Task Word Count Distribution")
        if not df['Word_Count'].empty:
            fig_word, ax_word = plt.subplots(figsize=(8, 4))
            sns.histplot(df['Word_Count'], bins='auto', kde=True, ax=ax_word, color='#00796b')
            ax_word.set_xlabel("Word Count in Task Description")
            ax_word.set_ylabel("Number of Tasks")
            ax_word.set_title("Distribution of Task Description Lengths")
            st.pyplot(fig_word)
        else:
            st.info("No word count data to display yet.")

        st.subheader("Workload by Category and Priority")
        workload_by_cat_prio = df.groupby(['Category', 'Priority'])['Workload_Score'].sum().unstack(fill_value=0)

        priority_order = ["Low", "Medium", "High"]
        existing_priority_columns = [col for col in priority_order if col in workload_by_cat_prio.columns]
        workload_by_cat_prio = workload_by_cat_prio[existing_priority_columns]

        if not workload_by_cat_prio.empty:
            fig_heat, ax_heat = plt.subplots(figsize=(8, 4))
            sns.heatmap(workload_by_cat_prio, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5, ax=ax_heat)
            ax_heat.set_title("Total Workload Score by Category and Priority")
            ax_heat.set_xlabel("Priority")
            ax_heat.set_ylabel("Category")
            st.pyplot(fig_heat)
        else:
            st.info("No combined category and priority workload to display yet.")

    else:
        st.info("Add some tasks on the 'Schedule Planner' page to generate detailed insights!")
