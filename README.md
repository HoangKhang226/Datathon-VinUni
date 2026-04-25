# Setup virtual environment
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Setup jupyter notebook
In PowerShell admin, run:
```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```
Then:
```powershell
.venv\Scripts\pip install jupyter ipykernel
```


