# Symanalytics
Symanalytics je nástroj pro analýzu dat s důrazem na jednoduchost použití a přístupnost pro české uživatele.
Program umožňuje provádění statistických testů či analýz. Nabízí několik klíčových funkcí:

- **Analýza rozptylu (ANOVA)**
- **Analýza závislosti v kontingenční tabulce**
- **Korelační analýza**
- **Lineární regrese**

### Hlavní výhody
- **Automatické testování předpokladů**: Program kontroluje, zda jsou splněny předpoklady pro jednotlivé statistické testy
- **Slovní interpretace výsledků**: Výsledky jsou prezentovány formou snadno srozumitelných slovních interpretací, což usnadňuje jejich pochopení

### Jazyková lokalizace
Celý program je vytvořen v češtině, což je výhodou pro české uživatele

### Podpora vstupních formátů
Symanalytics umožňuje nahrávání dat ve formátu Excel nebo CSV

### Technické specifikace
- **Programovací jazyk**: Python
- **Možnosti spuštění**: 
  - Přímo jako Python skript
  - Přímo jako EXE soubor pro Windows
  - Kompilace do EXE souboru pomocí 'pyinstaller'

# Návod na spuštění

## Stáhnout a spustit jako EXE soubor (Windows)
Nejjednoduší forma spuštění je stáhnout si EXE soubor pro Windows. Nachází se v záložce _Releases_ tohoto repozitáře. Není nutná žádná instalace programu ani není nutné mít Python v počítači, neboť celý program je zkompilovaný do jednoho spustitelného souboru. Jeho stažení či spuštění může být však obtížné, neboť prohlížeče automaticky blokují stahování neznámých EXE souborů a operační systém jejich spuštění. Proto může být nutné stažení a spuštění programu vícekrát potvrdit.

## Spustit jako Python skript
Pro spuštění je potřeba mít nainstalovaný Python spolu s využívanými balíčky.
Samotný program se spustí přes soubor "**Symanalytics.py**"

- **Potřebné standardní knihovny**:
  - tkinter
  - datetime
  - sys
- **Potřebné externí knihovny**:
  - numpy
  - pandas
  - scipy
  - matplotlib
  - openpyxl
  - xlrd

## Vytvořit vlastní EXE soubor
Je možné taky skript zkompilovat do EXE souboru ručně na vlastním počítači. Pro kompilaci je nutné mít nainstalovaný *Python* a knihovnu '*pyinstaller*'.

Poté stačí jen spustit připravený batch skript '*Compile_to_exe.bat*'. Celý program se následně automaticky zkompiluje do jednoho spustitelného souboru včetně všech potřebných knihoven. Výsledný soubor se bude následně nacházet v nově vytvořené složce *dist*.


# O vzniku programu
Program vznikl v roce 2024 v rámci bakalářské práce na Fakultě informatiky a statistiky Vysoké školy ekonomické. Celé znění práce včetně textové části je možné najít v [archivu závěrečných prací VŠE](https://vskp.vse.cz/92989_statisticke-metody-v-pythonu).

