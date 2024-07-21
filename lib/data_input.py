
print("Načítání modulu: Zpracování dat")

from lib import tools
from lib import memory

import tkinter as tk
from tkinter import filedialog
import pandas as pd




class DataInputSelector():
    def __init__(self, frame_left, frame_right, root = None) -> None:        
        """
        Creates button in frame_left that after clicking fill frame_right with data input options.

        frame_left: tkinter widget
            Parent left frame
        frame_right: tkinter widget
            Parent right frame
        root: tkinter widget
            Parent widget on which resize event will be called to properly resize widgets (fixes bug with resizing scrollbars)
        """

        self.frame_right = frame_right
        self.root = root

        # left panel button - data format
        tk.Button(frame_left, text="Nahrát soubor", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")

    def load_panel(self) -> None:
        
        precis = 3
        
        tools.destroy_all_children(self.frame_right)

        # choose file path
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Vyberte vstupní soubor s daty:", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        self.button_import = tk.Button(self.frame_right, text="Zvolit dataset", command=self.select_file, font=memory.fonts["text"], bg=memory.bg_color_button)
        self.button_import.pack(anchor="w")

        # selected file
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Zvolený soubor:", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        self.selected_path_label = tk.Label(self.frame_right, text=memory.file_source, font=memory.fonts["text"], bg=memory.bg_color_label)
        self.selected_path_label.pack(anchor="w")

        # file format (extention) option
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Zvolte formát souboru:", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        if memory.import_extention == None:
            memory.import_extention = tk.StringVar(None, "excel")
        tk.Radiobutton(self.frame_right, text="Excel (.xls / .xlsx)", variable=memory.import_extention, value="excel", command=self.import_extention_changed, font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Radiobutton(self.frame_right, text="CSV (.csv)", variable=memory.import_extention, value="csv", command=self.import_extention_changed, font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # include header option
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Další nastavení:", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        if memory.import_includes_header == None:
            memory.import_includes_header = tk.BooleanVar(None, True)
        tk.Checkbutton(self.frame_right, text="Obsahuje nadpis", onvalue=True, offvalue=False, variable=memory.import_includes_header, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).pack(anchor="w")
        
        # delimieter option
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        self.csv_delimieter_choises_buttons = []
        if memory.import_delimieter == None:
            memory.import_delimieter = tk.StringVar(None, ";")
        tk.Label(self.frame_right, text="Oddělovač (jen v případě CSV):", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        self.csv_delimieter_choises_buttons.append(tk.Radiobutton(self.frame_right, text="Čárka ( , )", variable=memory.import_delimieter, value=",", font=memory.fonts["text"], bg=memory.bg_color_radiobutton))
        self.csv_delimieter_choises_buttons.append(tk.Radiobutton(self.frame_right, text="Středník ( ; )", variable=memory.import_delimieter, value=";", font=memory.fonts["text"], bg=memory.bg_color_radiobutton))
        self.csv_delimieter_choises_buttons.append(tk.Radiobutton(self.frame_right, text="Mezera (   )", variable=memory.import_delimieter, value=" ", font=memory.fonts["text"], bg=memory.bg_color_radiobutton))
        self.csv_delimieter_choises_buttons.append(tk.Radiobutton(self.frame_right, text="Tečka ( . )", variable=memory.import_delimieter, value=".", font=memory.fonts["text"], bg=memory.bg_color_radiobutton))
        
        for button in self.csv_delimieter_choises_buttons:
            button.pack(anchor="w")
        self.import_extention_changed() # update buttons status (set initial status)

        # confirm changes
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Potvrdit nastavení (načíst data):", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        label_warning = tk.Label(self.frame_right, font=memory.fonts["text"], wraplength=300, justify="left", text="Pozor: Načtení souboru znamená ztrátu všech dříve provedených analýz a dalších nastavení!", bg=memory.bg_color_label)
        label_warning.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_warning)
        self.button_confirm = tk.Button(self.frame_right, text="Potvrdit nastavení", font=memory.fonts["text"], command=self.load_file, state="disabled", bg=memory.bg_color_button)
        self.button_confirm.pack(anchor="w")
        self.button_confirm_tooltip = tools.ToolTip(self.button_confirm, "Nejprve zvolte umístění vstupního souboru!")

        # table preview
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Dataset:", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Aktuálně nahraný dataset", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Ukázka dat je omezena na 15 řádků", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        
        #self.data_preview = tools.DataTable(memory.df, self.frame_right, self.root)
        frame_data_preview = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        frame_data_preview.pack(anchor="w")
        self.data_preview = tools.LabelsTable(frame_data_preview, memory.df, memory.fonts["header"], memory.fonts["text"], precis, row_limit=15)


        # eneble confirm changes button if file is already selected
        self.select_file(prompt_user=False)
        
        # show data preview if dataset selected
        if memory.dataset_selected == True:
            #self.data_preview.update_dataframe(memory.df, row_limit=15)
            self.data_preview.update_datafrme(memory.df)

        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # simulate initial resize event
        if self.root != None:
            tools.initial_resize_event(self.root)
        
    # set normal/disabled status of cvs delimieter based on if the csv file type is selected
    def import_extention_changed(self) -> None:
        """
        Based on variable "memory.import_extention" sets status (normal or disabled) to all radio buttons which are used to select delimieter for csv files.
        """
        for button in self.csv_delimieter_choises_buttons:
            if memory.import_extention.get() == "csv":
                button.configure(state="normal")
            else:
                button.configure(state="disabled")


    def select_file(self, prompt_user=True) -> None:
        """
        Prompts user to select file. The file directory is than saved into variable "memory.file_source" as a string.
        If some file is selected, variable "memory.dataset_selected" is than set to True, otherwise to False.

        prompt_user: Bool
            True: prompts user to select path to file
            False: only loads already selected path
        """
        if prompt_user:
            memory.file_source = filedialog.askopenfilename()
        
        if memory.file_source != "":
            memory.dataset_selected = True
            print("Byl zvolen datový soubor")
            self.selected_path_label.config(text = memory.file_source)
            self.button_confirm.config(state="normal")
            self.button_confirm_tooltip.tooltip_disable()
        else:
            memory.dataset_selected = False
            self.button_confirm.config(state="disabled")
            self.button_confirm_tooltip.tooltip_enable()
            if prompt_user:
                print("Nebyl zvolen datový soubor")
                #tools.error_message("Nebyl zvolen žádný soubor!")
                
    
    def load_file(self) -> None:
        """
        Loads file based on user selected settings and selected file path.
        """
        print("Načítání dat ze souboru...")
        
        match memory.import_includes_header.get(): # convert header settings from Bool (True/False) into valid argument for pandas read function (0/None)
            case True:
                include_header_argument = 0
            case False:
                include_header_argument = None
            case _:
                include_header_argument = None
                tools.error_message("Nastala neočekávaná chyba.\nNebylo zvoleno, jestli vstupní soubor obsahuje názvy sloupců. Zkuste změnit toto nastavení a akci opakujte.")
                return False

        match memory.import_extention.get():
            case "excel":
                print("Načítání souboru typu Excel")
                try: memory.df = pd.read_excel(memory.file_source, header=include_header_argument)
                except Exception as e:
                    self.load_file_error(e)
                    return False
            
            case "csv":
                print("Načítání souboru typu CSV")
                try: memory.df = pd.read_csv(memory.file_source, header=include_header_argument, delimiter=memory.import_delimieter.get())
                except Exception as e:
                    self.load_file_error(e)
                    return False
            
            case _:
                tools.error_message("Nebyla zvolen žádný typ souboru. Zkuste změnit nastavení typu souboru (CSV, Excel, ...) a akci opakujte.")
        
        # remove rows where all columns have nan value (source data might include empty rows for example in Excel)
        memory.df.dropna(how="all", inplace=True)
        
        # reset memory
        memory.reset_memory()

        print("\nNačtená data:")
        print(memory.df)
        #self.data_preview.update_dataframe(memory.df, row_limit=15)
        self.data_preview.update_datafrme(memory.df)

        for column in memory.df.columns:
            if column in memory.banned_column_name_list:
                tools.error_message(f"Zvolená proměnná '{column}' nesmí být použitá v jakékoliv analýze, musí být nejprve přejmenovaná. Tento název sloupce je vyhrazen pouze pro interní výpočty v programu. Využijte prosím nastavenní proměnných pro přejmenování.\n{memory.banned_columns_string}")


        if self.root != None:
            tools.initial_resize_event(self.root)
    
    def load_file_error(self, exception) -> None:
        error_message = "Nastala neočekávaná chyba při načítání souboru. Ujistěte se, že jste zvolili správný typ souboru (CSV, Excel, ...) a že je zvolený správný soubor pro načtení.\n\nChybová hláška:\n\n"
        error_message = error_message + str(exception)
        tools.error_message(error_message)



class DataInputFormater():
    
    def __init__(self, frame_left, frame_right, root=None) -> None:        
        """
        Creates button in frame_left that after clicking fill frame_right with data format options.

        frame_left: tkinter widget
            Parent left frame
        frame_right: tkinter widget
            Parent right frame
        root: tkinter widget
            Parent widget on which resize event will be called to properly resize widgets (fixes bug with resizing scrollbars)
        """
        
        self.frame_right = frame_right
        self.root = root

        # left panel button - data format
        tk.Button(frame_left, text="Nastavení proměnných", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")
    
    def load_panel(self) -> None:
        
        precis = 3
        
        tools.destroy_all_children(self.frame_right)
        tk.Label(self.frame_right, text="Nastavení proměnných (sloupců)", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        
        self.settings_frame = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.settings_frame.pack(anchor="w")

        tk.Label(self.settings_frame, text="Současný název   ", font=memory.fonts["header"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        tk.Label(self.settings_frame, text="Nový název   ", font=memory.fonts["header"], bg=memory.bg_color_label).grid(row=0, column=1, sticky="w")
        tk.Label(self.settings_frame, text="Typ proměnné   ", font=memory.fonts["header"], bg=memory.bg_color_label).grid(row=0, column=2, sticky="w")

        self.column_names_var = []
        self.column_types_var = []
        
        for i, column in enumerate(memory.df.columns):
            tk.Label(self.settings_frame, text=str(column), font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=i+1, column=0, sticky="w")
            
            # Entry - new column name
            self.column_names_var.append(tk.StringVar(value=str(column)))
            tk.Entry(self.settings_frame, font=memory.fonts["text"], textvariable=self.column_names_var[i], bg=memory.bg_color_text).grid(row=i+1, column=1, sticky="w")

            # Dropdown - new datatype
            self.column_types_var.append(tk.StringVar(value=tools.convert_data_types_to_human(str(memory.df[column].dtype))))
            tk.OptionMenu(self.settings_frame, self.column_types_var[i], *memory.datatype_options).grid(row=i+1, column=2, sticky="w")
        
        # save changes button
        tk.Button(self.frame_right, text="Uložit změny", command=self.save_changes, font=memory.fonts["text"], bg=memory.bg_color_button).pack(anchor="w")
        
        # data preview
        self.preview_frame = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.preview_frame.pack(anchor="w")
        tk.Label(self.preview_frame, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.preview_frame, text="Ukázka dat", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        #self.data_preview = tools.DataTable(memory.df, self.preview_frame, self.root)
        
        frame_preview_table = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        frame_preview_table.pack(anchor="w")
        self.data_preview = tools.LabelsTable(frame_preview_table, memory.df, memory.fonts["header"], memory.fonts["text"], precis, row_limit=20)
        
        #if memory.dataset_selected == True:
        #    self.data_preview.update_dataframe(memory.df, row_limit=15)

        
        
        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)


    def save_changes(self) -> None:
        


        # check for duplicate names of columns
        column_names_string = []
        for i, col_name in enumerate(self.column_names_var):
            column_names_string.append(col_name.get())
        
        if len(column_names_string) != len(set(column_names_string)):
            tools.error_message("Jména jednotlivých sloupců musí být zcela unikátní. Prosím změňte názvy sloupců.")
            return
        
        for i, col_name in enumerate(self.column_names_var):
            memory.df.rename(columns={memory.df.columns[i]: col_name.get()}, inplace=True)



        error_columns = []
        
        for i, col_type in enumerate(self.column_types_var):
            
            col_type = tools.convert_data_types_to_code(col_type.get())
            try:
                memory.df.iloc[:, i] = memory.df.iloc[:, i].astype(col_type)
            except Exception as e:
                print(f"Chyba při změně datového typu sloupce {i+1}:\n\n{e}")
                error_columns.append(i)
        
        if len(error_columns) > 0:
            error_string = ""
            for col_index in error_columns:
                error_string = error_string + f"Sloupec {col_index+1}: {column_names_string[col_index]}\n"
            tools.error_message(f"Došlo k chybě při změně datového formátu u následujících sloupců:\n\n{error_string}")
        
        print("\nDataset byl upraven")
        print("Upravený dataset")
        print(memory.df)
        print("Datové typy proměnných:")
        print(memory.df.dtypes)

        memory.reset_memory()

        self.load_panel()
