

@echo :::::::::::::::::::::::::::::::::::::::::::::
@echo Kompilace kodu do spustitelného EXE souboru
@echo :::::::::::::::::::::::::::::::::::::::::::::

@echo.

@echo Spustenim tohoto skriptu se Python kod zkompiluje do spustitelneho EXE soubor vcetne vsech potrebnych knihoven
@echo Vysledny EXE soubor se bude nachazet ve slozce .\dist\...
@echo Pro zobrazeni spravne ikony vysledneho zkompilovaneho exe souboru je nutne udelat kopii exe souboru (bug ve Windows pro zobrazovani ikon)
@echo.
@echo Pro kompilaci souboru je nutne mit nainstalovanou Python knihovnu 'pyinstaller'

@pause

:: pyinstaller .\Symanalytics.py
:: pyinstaller --onefile --hidden-import=xldr --hidden-import=openpyxl .\Symanalytics.py
:: pyinstaller --onefile --hidden-import=xldr --hidden-import=openpyxl --icon=Icon.ico .\Symanalytics.py
pyinstaller --onefile --hidden-import=xldr --hidden-import=openpyxl --icon=Icon.ico --splash "Splash_loading.png" .\Symanalytics.py

@echo Vysledny EXE soubor se bude nachazet ve slozce \dist\
@echo Pro zobrazeni spravne ikony vysledneho zkompilovaneho exe souboru je nutne udelat kopii exe souboru (bug ve Windows pro zobrazovani ikon)

@pause