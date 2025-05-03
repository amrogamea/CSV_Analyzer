import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
import os
import chardet
import streamlit as st

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["api"]["openai_api_key"]

# Set CSV file path
CSV_FILE_PATH = '/Users/ahmedsalah/Projects/Chat with CSV/Full MV & CC data - Fixed.csv'

class CSVAnalyzer:
    def __init__(self):
        """Initialize the CSV analyzer."""
        self.memory = ConversationBufferMemory()
        self.df = None
        self.agent = None

    def clear_memory(self):
        """Clear the conversation memory/history."""
        self.memory.clear()
        return "Memory has been cleared. Conversation history has been reset."

    def detect_encoding(self, file_path):
        """Detect the encoding of a file."""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']

    def preprocess_dataframe(self, df):
        """Preprocess the DataFrame to handle common data cleaning tasks."""
        df_processed = df.copy()

        # Convert date columns
        for col in df_processed.columns:
            if any(word in col.lower() for word in ['date', 'time', 'day', 'month', 'year']):
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col])
                except:
                    pass

        df_processed['ActionDate'] = pd.to_datetime(df_processed['ActionDate'], dayfirst=True, errors='coerce')
        
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                try:
                    sample = df_processed[col].dropna()
                    if not sample.empty:
                        if all(str(val).replace('.', '', 1).replace('-', '', 1).isdigit() for val in sample.head(5)):
                            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                except:
                    pass

        if 'SubAction' in df_processed.columns and 'ActionType' in df_processed.columns:
            switch_mask = df_processed['SubAction'].str.contains('switch', case=False, na=False)
            df_processed.loc[switch_mask, 'ActionType'] = 'Placements'

        return df_processed

    def generate_metadata(self, df):
        """Generate comprehensive metadata about the DataFrame."""
        date_range = None
        if 'ActionDate' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ActionDate']):
            date_range = {
                'start': df['ActionDate'].min().strftime('%Y-%m-%d'),
                'end': df['ActionDate'].max().strftime('%Y-%m-%d')
            }
        
        return {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "date_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
            "subaction_unique_values": df['SubAction'].unique().tolist() if 'SubAction' in df.columns else [],
            "date_range": date_range
        }

    def create_metadata_prompt(self, metadata):
        """Create a system prompt that includes the metadata."""
        prompt = f"""You are a data analysis assistant working with a DataFrame. Here is the metadata about the data:

Columns: {metadata['columns']}
Data Types: {metadata['dtypes']}
Number of rows: {metadata['shape'][0]}
Number of columns: {metadata['shape'][1]}

Numeric columns: {metadata['numeric_columns']}
Categorical columns: {metadata['categorical_columns']}
Date columns: {metadata['date_columns']}

Please use this information to provide accurate analysis and avoid making assumptions about data that doesn't exist."""
        return prompt

    def load_csv(self, file_path):
        """Load and process a CSV file."""
        try:
            # Detect file encoding
            encoding = self.detect_encoding(file_path)
            
            # Read the CSV file
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Preprocess the DataFrame
            self.df = self.preprocess_dataframe(df)
            
            # Generate metadata
            metadata = self.generate_metadata(self.df)
            system_prompt = self.create_metadata_prompt(metadata)

            # Create the agent
            prefix = f"""
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Consider any question about sales as asking about "Placements"
The following list is a dictionary of technical terms . For each term, there are alternative words or phrases that users may use instead of the given technical term. These are grouped under each term to ensure that you can understand and interpret questions correctly, even if users employ different language or terminology.
Dictionary Section:
Placements:
Other terms: Sales, New Sales,Contract Signed, CLIENT SALES.
Cancellations Live-in:
Other terms: Live-in Contract Canceled, live in contract terminated 
Cancellations Live-out:
Other terms: Live-out Contract Canceled, live out contract terminated 
Maid Recruitment:
Other terms: Maid Hiring, recruited, Maid Employment, joined, landed in dubai. 
Maid Attritions:
Other terms: Maid Terminated, Maid Fired, Maid Resigned, Non Renewal maids.

<DATA EXPLANATION>

1- Column "ActionType" gives the type of the main action that we are tracking. we have 5 main action types: [Placements (which stands for sales or new contracts), Cancellations, Replacement, Maid Recruitment, Maid Attritions(stands for maids terminated and the breakdown of the termination lies in the "SubAction" column)]
2- Column "ActionDate" gives the date of the action.
3- Column "Nationality" gives the nationality of the Maid associated with the action.
4- Column "Prospect Type" let us know if this action is an MV or a CC (Live-in or Live-out or both) Action, you must use this column when the query is about MV or CC( live-in or live-out).
5- Column "IDType" shows weather it's a "Contract" id or a "Maid" id. 
6- Column "ID" is the id of the contract or the maid. use that when searching for a specific contract(s) or maid(s) among different groups.
7- Column "SubAction" gives the type of the sub action that we are tracking. you must check the values of this column when you need can't map the input query with a certain main action.
8- Column "SubAction" has the following unique values that you must be aware of: {metadata['subaction_unique_values']}
9- Never filter the sub action column if the query doesn't imply a breakdown from the main action. (e.g. if the query is about the total number of placements, don't filter the sub action column)
10- The full date range of this dataset is [ {metadata['date_range']['start']} to {metadata['date_range']['end']} ]. if the user asks about a date range that is not in this range, you must inform them and ask for a date range in the valid range.

</DATA EXPLANATION>
Rules:
1- For any Date inquiry, Always use "ActionDate" column because it's a datetime object using day first format.
2- Always differentiate between questions about Replacements and the ones about Returning Maids, Replacements is a main action type and means that the client still have contract with us, while "Returning Maids" is a sub action type and associated with a main action type of "Cancellations" and means that the client doesn't have any contract with us. make sure to extract the intent of the user correctly from the query.
3- for questins about cancellations, never refrence the sub action column without looking in all it's unique values in the "SubAction" column and match the input query with the most likely sub action.
4-  When asked about cancellations, Always try to provide breakdowns for every subcategory associated with the cancellation while also showing the total in your final answer. this is very important.
5- When a question contains a structure that suggests a breakdown or a comparison like "Out of them, how many...", understand it as a two-step filtering process that you must use the contract id and merging techniques based on the contract id to find the answer smoothly:
1. First, apply the initial condition to get a subset of the data (e.g., all African sales in 2024).
2. Then, use this subset as the basis for a second condition (e.g., from those, how many were returned or canceled) using merging based on contract id between the subset and the entire dataframe.
3. Merging will make things easier and more accurate in terms of coding and filtering.

Do not reset or re-filter the entire dataset in step 2 — always apply the second condition only to the subset obtained from step 1. Always use merging using contract id when possible.

Ensure that any references like "them" or "those" point to the previously filtered group. Use contract id always to check further conditions, like Cancellations or Replacements.

6- When asked about CC sales or CC cancellations without specifying it's a live-in or live-out, always assume that it means both categories (Live-in and Live-out) unless otherwise specified, always use the "Prospect Type" column to do such filtering.
7- When asked about MV use the "Prospect Type" column to do such filtering and find the MV actions, MV and CC(live-in and live-out) are different actions.
8- MV & CC can be written in lower case or upper case.
You should use the tools below to answer the question posed of you: """

            suffix = f"""
This is the result of print(df.head()):
{{df_head}}

**Important instruction:**
Before answering, carefully check if the user's question is clear and complete.
- If it lacks necessary details (e.g., only month is given but no year mentioned, full date range missing, category names missing), ask a clarifying question first. 
- Never assume dates (year, month or day) and always ask for them if not provided.
- Otherwise, proceed to answer.

**IMPORTANT Rule you must always follow to avoid runtime errors**:
Every <Thought:> must immediately be followed by either:
<Action:> AND <Action Input:>, or <Final Answer:>

- Do NOT produce both an <Action> and a <Final Answer> at the same time.
- If clarification from the user is needed, you must output it as a <Final Answer:> asking your clarification question. this must be doen every
- Do not output any text outside <Thought:>, <Action:>, <Action Input:>, and <Final Answer:> blocks.
**End of the IMPORTANT Rule**

Conversation history:
{{history}}

Begin!
Question: {{input}}
{{agent_scratchpad}}"""

            self.agent = create_pandas_dataframe_agent(
                llm=ChatOpenAI(model="gpt-4-turbo", temperature=0.1),
                df=self.df,
                prefix=prefix,
                suffix=suffix,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                include_df_in_prompt=None,
                handle_parsing_errors=True
            )
            
            return True
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return False

    def ask_question(self, question):
        """Ask a question about the loaded CSV data."""
        if self.agent is None:
            return "Please load a CSV file first using load_csv()"
        
        try:
            response = self.agent.run(
                input=question,
                history=self.memory.buffer
            )
            self.memory.save_context(
                {"input": question},
                {"output": response}
            )
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"

def main():
    """Main function to run the CSV analyzer in console mode."""
    analyzer = CSVAnalyzer()
    
    if not analyzer.load_csv(CSV_FILE_PATH):
        print("Failed to load CSV file. Exiting...")
        return
    
    print("\nCSV file loaded successfully! You can now ask questions about your data.")
    print("Type 'exit' to quit.\n")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
            
        response = analyzer.ask_question(question)
        print("\nResponse:", response)

if __name__ == "__main__":
    main() 