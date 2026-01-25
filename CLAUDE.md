# Project description

The project's aim is to train an encoder decoder transformer to generate piano pedal information given note context. The pedal information is simplyfied via the segment class (which describes a piece of function)
The project has a customly made transformer approach in codebase and a pretrained and finetune approach with midiBert.

# Project Overview (relevant files)

Project Path: `C:\Users\edgar\Documents\Studium\Mathe\Bachelor\Code`

- codebase (folder)
   - `miscellaneous` (folder)
   - `data.py`
   - `inference.py`
   - `model.py`
   - `preprocessing.py`
   - `train.py`
   - `utils.py`
   - `evaluate.py`

- Orga (folder)
    - LucidChart.png - Overview of the entire project in UML diagram. Best way to get overwiev of the project.
    - Notizen.txt - For User. Todo's.
    - Prompt.txt - Original prompt to create this project

- saves (folder)
- ML_With_Torch_venv (folder) - 
- main_codebase.ipynb and main_midiBERT.ipynb- where user runs and tests code from
- main.py - For Claude Code to run and debug main code. Always keep it very minimal so user quickly sees what Claude is up to

# How to run code
cd "C:/Users/edgar/Documents/Studium/Mathe/Bachelor/Code" && ML_With_Torch_venv/Scripts/python.exe example_script.py
**IMPORTANT: ALWAYS EXECUTE CODE FROM THIS VIRTUAL ENVIRONMENT**

# Development Guidelines

## Git usage
- You are **STRICTLY FORBITTEN** to run git checkout <file_name>, git reset and git restore - all commands that can potentially delete progress.

## Let code fail
- Don't failproof the code.
- ALWAYS ASSUME THE CODE DOES EXACTLY WHAT IT'S SUPPOSED TO. If it doesn't, the user wants to know immediately when and where it failed
- **NO TRY & CATCH BLOCKS** or any other safeguards
- No checking for empty or short inputs. 
- **NO CLAMPING** to avoid accidental division by zero

## Debugging
- Don't create corrected functions with modified names (e.g., `processor_corrected`). Instead, keep original function names and improve the code.
- **No backward compatability**
- Always look for code that can be deleted because of changes made
- Always try to simplify code
- Add only crucial comments necessary for understanding the code
- name test scripts test_[what_you_test]

## Project specific rules
- all save and loading functions append "saves/" to their path by default
- all dataset files are saved in stream of multiple tuples format