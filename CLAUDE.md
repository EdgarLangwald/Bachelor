# Project description

The project's aim is to train an encoder decoder transformer to generate piano pedal information given note context. The pedal information is simplyfied via the segment class (which describes a piece of function)
The project has a customly made transformer approach in codebase and a pretrained and finetune approach with midiBert.

# Project Overview (relevant files)

Project Path: `C:\Users\edgar\Documents\Studium\Mathe\Bachelor\Code`

- codebase (folder)
   - `data.py`
   - `inference.py`
   - `model.py`
   - `preprocessing.py`
   - `train.py`
   - `utils.py`

- midiBERT
   - `data.py`
   - `inference.py`
   - `midibert_wrapper.py`
   - `model.py`
   - `preprocessing.py`
   - `tokenization.py`
   - `train.py`
   - `utils.py`

- Orga (folder)
    - LucidChart.png - Overview of the entire project in UML diagram. Best way to get overwiev of the project.
    - Notizen.txt - For User. Todo's.
    - Prompt.txt - Original prompt to create this project

- saves (folder)
- ML_With_Torch_venv (folder) - 
**IMPORTANT**: always execute code from this venv
- main_codebase.ipynb and main_midiBERT.ipynb- where user runs and tests code from
- main.py - For Claude Code to use to run and debug main code. Always keep it very minimal so user quickly sees what Claude is testing

# Development Guidelines

- **ADHERE TO THESE GUIDELINES**
- You are **STRICTLY FORBITTEN** to run git checkout <file_name>, git reset and git restore - all commands that can potentially delete progress.
- Don't failproof the code. ALWAYS ASSUME THE CODE DOES EXACTLY WHAT IT'S SUPPOSED TO. If it doesn't, the user wants to know immediately when and where it failed. NO TRY & CATCH BLOCKS or any other safeguards. No checking for empty or short inputs. Asserts are fine. 
When Debugging:
- GOAL: KEEP CODE AS SLIM AS POSSIBLE. this means:
- Don't create corrected functions with modified names (e.g., `processor_corrected`). Instead, keep original function names and improve the code.
- Never implement backward compatability
- Always look for code that can be deleted because of changes made
- Always try to simplify code
- Add only crucial comments necessary for understanding the code
- name test scripts test_[what_you_test]