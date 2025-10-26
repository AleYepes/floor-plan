# floor-plan
Script to calculate forces for new floor in my basement

# Requirements

## Install nbstripout to Remove Unnecessary Metadata and Use nbdime for Better Diffs and Merging
```
pip install nbstripout nbdime
nbstripout --install
nbdime config-git --enable
```

To manually compare two notebook versions:
```
nbdiff notebook_1.ipynb notebook_2.ipynb
```

To resolve conflicts interactively:

```
nbmerge notebook.ipynb
```