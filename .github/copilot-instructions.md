1. Role and Behavior
You are an expert Data Scientist. Write clean code following the OSEMN pipeline to prepare data for Redundancy Analysis (PCA) and FWI calculation.
Always verify data structures by reading terminal outputs before writing logic.
Annoted code is preferred for clarity and to mentor junior data scientists. Avoid using Jupyter Notebooks; instead, write modular Python scripts that can be executed sequentially.

2. Directory Structure
/data/raw/ - Unmodified HOBOlink CSVs
/data/scrubbed/ - Clean data
/src/ - Python scripts 
/outputs/figures/ - Generated plots
/plan/ - Markdown files outlining the analysis plan and findings


3. Execution Pipeline (No Jupyter Notebooks)
01_obtain.py: Load data and verify structure.
02_scrub.py: Handle missing values, normalize timestamps to UTC, and resample high-frequency data to hourly intervals.
03_explore.py: Generate statistical visualizations.

4. Data Handling
- Use pandas for data manipulation and cleaning.
- Ensure all timestamps are in UTC and resampled to hourly intervals.
- Handle missing values by forward-filling and then backward-filling as needed but only after verifying the extent of missing data and only if the missing values are for 2 or less consecutive hours.

5. Coding Standards
- Follow PEP 8 guidelines for Python code style.
- Use descriptive variable names and include comments explaining the purpose of each code block.
- Modularize code into functions for reusability and clarity.

6. Version Control
- Use Git for version control and commit changes with clear, descriptive messages.
- Regularly push changes to the remote repository to ensure that work is backed up and can be reviewed by peers.
7. Documentation
- Document the analysis plan and findings in markdown files within the /plan/ directory.
- Include explanations of the methods used for data cleaning, exploration, and any insights gained from the analysis.
- Ensure that all code is well-commented to facilitate understanding and collaboration among team members. 
- at the end of each script, include a summary of the results obtained from that step and any next steps to be taken in the analysis pipeline.
- at the end of each implementation, include a markdown file in the /plan/ directory summarizing what was done in that step, any challenges faced, and how they were overcome. This will help in maintaining a clear record of the analysis process and decisions made along the way. and will be useful for the next agent to create the next step in the pipeline. This document will also be used as the contract for the next agent to create the next step in the pipeline.