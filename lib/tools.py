
print("Načítání modulu: Nástroje")

from lib import memory

import tkinter as tk
from tkinter.ttk import Treeview
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk



def get_datetime_string():
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%d/%m/%y - %H:%M:%S")
    return formatted_datetime

def standardize_vector(vector: np.array) -> np.array:
    """Takes numpy array and performs standardization. Returns standardized array.
    Formula:
        (x_i - mean) / standard deviation
    Args:
        vector (np.array): vector of input values
    Returns:
        np.array: vector of standardized values
    """
    n = len(vector)
    mean = np.mean(vector)
    sd = np.sqrt(
        np.sum((vector - mean) ** 2) / n
    )

    return (vector - mean) / sd



def error_message(message, title="Chyba") -> None:
    print(f"---------- Chyba: {title} ----------\n{message}\n------------------------------")
    tk.messagebox.showwarning(title=title, message=message, icon="error")

def clear_right_panel(main_panned_window) -> None:
    """
    Destoys 2nd children on inputed widget. It is designed to clear right panle if the parrent of the frame is inputed.
    """
    main_panned_window.winfo_children()[1].destroy()

def destroy_all_children(parent_widget: tk.Frame) -> None:
    for child in parent_widget.winfo_children():
        child.destroy()

    if hasattr(parent_widget, "wrappable_labels"):
        parent_widget.wrappable_labels = []


def enable_disable_widget(boolean, widget):
    if boolean:
        widget.config(state="normal")
    else:
        widget.config(state="disabled")

def initial_resize_event(widget) -> None:
    """
    Simulates 2x configure event to widget. Use this command at the end of code when filling panel with new data (to update scroll bars).
    It is best to use it for 'root', but could be potentialy used for any frame.
    """
    for i in range(2): # must be done 2x to work properly
        widget.event_generate("<Configure>")


def long_to_short(df_long: pd.DataFrame, col_categories: str, col_values: str, debug_print=False) -> pd.DataFrame:
    """
    Transforms pandas Dataframe from long format to short format.
    Which means that values in value column are devided into separate columns based on value in categorical column.
    
    Parameters
        df_long: pandas DataFrame
            Pandas dataframe in long format
        col_categories: str
            Name of column with categorical data
        col_values: str
            Name of column with values
        debug_print: bool
            Whether dataframe should be printed to console
    Returns
        Returns pandas Dataframe in short format
    """
    # iterate through unique classes and save columns as separate dataframes
    dataframe_list = []
    unique_categories = df_long[col_categories].unique()
    
    for category in unique_categories:
        values = df_long[[col_values]][df_long[col_categories] == category]
        values.columns = [category]
        values.reset_index(inplace=True, drop=True)
        dataframe_list.append(values)
    
    # create a new DataFrame by concatoning columns
    # it has to be complicated like this in order to avoid issues with different lengths of columns
    df_short = pd.DataFrame()
    for column_df in dataframe_list:
        df_short = pd.concat([df_short, column_df], axis=1)

    if debug_print:
        print("DataFrame long to short:")
        print(df_short)
    return df_short


def create_scrollable_frame(parent_panel, stretch_width=False):
    """
    Creates scrollable frame.
    Also includes attribute list 'wrappable_labels'. Labels appended to that list are dynamically wrapped based on width of the frame.
    
    Do not forget to use following command to set initial wrap length:
    'self.right_panel_name.event_generate("<Configure>", width=0, height=0)'
    It generates fake resize event to the parent widget which than fires up resize event for all its childern which forces initial wrapping length to be set.
    
    For scrollable function code inspired by https://blog.teclado.com/tkinter-scrollable-frames/
    """
    frame_main_container = tk.Frame(parent_panel, bg=memory.bg_color_scroll_container)
    frame_main_canvas = tk.Canvas(frame_main_container, bg=memory.bg_color_scroll_canvas)
    frame_main_scrollbar_y = tk.Scrollbar(frame_main_container, orient="vertical", command=frame_main_canvas.yview)
    frame_main_scrollbar_x = tk.Scrollbar(frame_main_container, orient="horizontal", command=frame_main_canvas.xview)

    frame_main = tk.Frame(frame_main_canvas, width=1000, bg=memory.bg_color_scroll_frame)
    
    frame_main.wrappable_labels = [] # list of lablels to by dynamically wrapped (sets wrap length)

    window_ID = frame_main_canvas.create_window((0, 0), window=frame_main, anchor="nw")

    def change_maximum_width_of_text_widgets() -> None:
        for text_widget in frame_main.wrappable_labels:
            text_widget.config(wraplength = int(max(100, frame_main_canvas.winfo_width()-20)))
    
    def update_size_scroll(event) -> None:
        frame_main_canvas.configure(scrollregion=frame_main_canvas.bbox("all"))
        #frame_main_canvas.itemconfig(window_ID, height=frame_main_canvas.winfo_height())
        #if stretch_width:
            #frame_main_canvas.itemconfig(window_ID, width=frame_main_canvas.winfo_width())
        change_maximum_width_of_text_widgets()

    frame_main_container.bind("<Configure>", update_size_scroll)
    
    frame_main_canvas.configure(yscrollcommand=frame_main_scrollbar_y.set, xscrollcommand=frame_main_scrollbar_x.set)

    parent_panel.add(frame_main_container)

    frame_main_scrollbar_y.pack(side="right", fill="y")
    frame_main_scrollbar_x.pack(side="bottom", fill="x")
    frame_main_canvas.pack(side="left", fill="both", expand=True)

    frame_main_container.update()

    update_size_scroll(None)
        
    
    return frame_main




class ToolTip:
    """
    Creates tooltip that shows up when mouse is hovering over the widget

    Insperation: https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
    Used: Chat GPT
    """
    def __init__(self, widget, text, enabled = True):
        self.widget = widget
        self.text = text
        self.tooltip_enabled = enabled
        self.tooltip = None
        self.bind_enter = self.widget.bind("<Enter>", self.show_tooltip)
        self.bind_leave = self.widget.bind("<Leave>", self.hide_tooltip)
    
    def tooltip_enable(self):
        if self.tooltip_enabled == False:
            self.bind_enter = self.widget.bind("<Enter>", self.show_tooltip)
            self.bind_leave = self.widget.bind("<Leave>", self.hide_tooltip)
            self.tooltip_enabled = True
    
    def tooltip_disable(self):
        if self.tooltip_enabled:
            self.hide_tooltip()
            self.widget.unbind("<Enter>", self.bind_enter)
            self.widget.unbind("<Leave>", self.bind_leave)
            self.tooltip_enabled = False

    def show_tooltip(self, event = None):
        x = y = 0
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text, background="#ededed", relief="solid")
        label.pack()

    def hide_tooltip(self, event = None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class DataTable():
    """
    Creates table that can show data from pandas dataframe.
    To pack and view table, function 'update_dataframe' must be executed.

    data_frame: pandas.DataFrame
        input dataframe
    parent_widget: tkinter widget
        parent widget
    root: tkinter widget
        main tkinter parent that contains scrollbars
        optional, but recommended for correct scrollbar function
    print_debug: bool
        if debug messages should be printed to console

    """
    def __init__(self, data_frame, parent_widget, root=None, print_debug=False) -> None:
        
        self.parent_widget = parent_widget
        self.data_frame = data_frame
        self.root = root
        self.print_debug = print_debug
        self.tree = None

    def kill_table(self):
        if self.tree != None:
            self.tree.pack_forget()
            self.tree.destroy()
            self.tree = None

    def update_dataframe(self, data_frame: pd.DataFrame, row_limit = None) -> None:
       
        self.kill_table()

        self.columns = data_frame.columns.to_list()
        if self.print_debug:
            print("columns:")
            print(self.columns)

        self.tree = Treeview(self.parent_widget, columns=self.columns)
        self.tree['show'] = 'headings'

        # Add headings (column names) to the treeview
        for index, col in enumerate(self.columns):
            self.tree.heading(f"#{index+1}", text=col)

        # Insert data from the DataFrame into the treeview
        if self.print_debug:
            print("rows:")
        for index, row in data_frame.iterrows():
            if row_limit != None: # limit data preview
                if index > row_limit: 
                    if self.print_debug:
                        print("row limit reached")
                    break
            if self.print_debug:
                print(row.tolist())
            self.tree.insert('', 'end', values=row.tolist())
        
        self.tree.configure(height=len(data_frame.index))
        self.tree.pack(anchor="w", fill="both")

        initial_resize_event(self.parent_widget)

        if self.root != None:
            initial_resize_event(self.root)


def convert_data_types_to_human(data_type_str):
    """
    Convert Pandas data type strings to user-friendly names.
    data_type_str (str): The Pandas data type string to be converted.
    Returns: str: The user-friendly name for the data type.
    Example:
        convert_data_types('object')
        'Kategorie (text)'
        convert_data_types('int64')
        'Celá čísla'
    """
    
    data_type_mapping = {
        'object': 'Kategorie (text)',
        'str': 'Kategorie (text)',
        'category': 'Kategorie (text)',
        'int64': 'Celá čísla',
        'int32': 'Celá čísla',
        'float64': 'Desetinná čísla',
        'float32': 'Desetinná čísla',
        'datetime64[ns]': 'Datum a čas',
        'bool': 'Logická hodnota (ano/ne)',
    }

    result = data_type_mapping.get(data_type_str, "nepodporovaný typ proměnné")
    if result == "nepodporovaný typ proměnné":
        print(f"Chyba: Datový typ '{data_type_str}' není podporovaný")
    return result


def convert_data_types_to_code(data_type_str):
    """
    Convert user-friendly data type names to their corresponding Python code representations.
    data_type_str (str): The user-friendly data type name to be converted to code representation.
    Returns: type or None: The Python code representation of the data type or None if the input is not recognized.
    """
    
    data_type_mapping = {
        'Kategorie (text)': str,
        'Celá čísla': int,
        'Desetinná čísla': float,
        'Datum a čas': np.datetime64,
        'Logická hodnota (ano/ne)': bool,
    }
    result = data_type_mapping.get(data_type_str)
    if result == None:
        print(f"Chyba: Datový typ '{data_type_str}' není podporovaný")
    return result

def check_variable(df: pd.DataFrame, column: str, require_number: bool, unique_value_limit:int=None, number_values_limit:int=None) -> bool:
    """Checks if selected column is in dataframe. Can check if column is numerical. Can check for number of unique values in column (for example if there is too many categories). Can check for number of values.

    Args:
        df (pd.DataFrame): dataframe with the columns
        column (str): selected column from dataframe
        require_number (bool): wether variable must be numerical (or compatible like bool)
        unique_value_limit (int, optional): maximum number of unique values in column. Defaults to None.
        number_values_limit (int, optional): maximum number of values in column. Defaults to None.

    Returns:
        bool: _description_
    """

    if column not in df.columns:
        error_message(f"Zvolená proměnná '{column}' není v nahraném datasetu. Prosím, znovu nastavte zvolené proměnné. V případě opakované chyby program restartujte.")
        return False
    
    if column in memory.banned_column_name_list:
        error_message(f"Zvolená proměnná '{column}' musí být nejprve přejmenovaná. Tento název sloupce je vyhrazen pouze pro interní výpočty v programu. Využijte prosím nastavenní proměnných pro přejmenování.\n{memory.banned_columns_string}")
        return False
    
    if require_number == None and unique_value_limit == None and number_values_limit == None:
        return True
    


    col_values = df[column]
    col_type = str(col_values.dtype)


    if require_number:
        if col_type not in ("int64", "int32", "float64", "float32", "bool"):
            error_message(f"Zvolená proměnná '{column}' není číselná.\nProměnná je typu '{col_type}' ({convert_data_types_to_human(col_type)}).\n\nUjistěte se, že jste zvolili správnou proměnnou.\n\nPřípadně využijte nastavení proměnných, kde je možné proměnnou převést na číselnou, jesltiže pro to splňuje předpoklady.")
            return False
    
    if unique_value_limit != None:
        unique_values = len(col_values.unique())
        if unique_values > unique_value_limit:

            response = tk.messagebox.askquestion("Kontrola nastavení", f"Zvolená proměnná '{column}' přesáhla doporučené množství unikátních hodnot.\nProsím ujistěte se, že je zvolená správná proměnná, než budete pokračovat.\n\nPočet unikátních hodnot v proměnné: {unique_values}\nDoporučené maximum unikátních hodnot: {unique_value_limit}\n\nPřekročení doporučeného počtu hodnot může způsobit neočekávané chyby. Přejete si přesto pokračovat?", icon='warning')
            if response == "yes":
                print(f"Proměnná '{col_values}' přesáhla doporučený počet unikátních hodnot - pokračování ve výpočtu")
                return True
            else:
                print(f"Proměnná '{col_values}' přesáhla doporučený počet unikátních hodnot - zastavení výpočtu")
                return False
        
    return True




class LabelsTable():
    def __init__(self, parent_frame: tk.Frame, dataframe: pd.DataFrame, font_header, font_text, round_values=None, first_col_allign_left=False, print_debug=False, row_limit=None) -> None:
        """Creates table from lables
        Args:
            parent_frame (tk.Frame): Parent frame widget that will be used for this table only and will use grid inside
            DataFrame (pd.DataFrame): Pandas dataframe with values and named headers
            font_header:
            font_text:
            round_values: None if values shloud not be rounded, integer to set the number of decimals to round to
            row_limit: None if whole table shloud be printed, integer to set maximum number of rows that will be shown
        """
        
        self.parent_frame = parent_frame
        self.print_debug = print_debug
        self.round_values = round_values
        self.first_col_allign_left = first_col_allign_left
        self.font_header = font_header
        self.font_text = font_text
        self.row_limit = row_limit


        self.update_datafrme(dataframe)
        
    
    def update_datafrme(self, dataframe: pd.DataFrame) -> None:
        """Updates table from lables
        Args:
            parent_frame (tk.Frame): Parent frame widget that will use grid inside
            DataFrame (pd.DataFrame): Pandas dataframe with values and named headers
            font_header
            font_text
            round_values: None if values shloud not be rounded, integer to set the number of decimals to round to
        """

        if self.print_debug:
            print("\n-- LabelsTable --\n")
            print(dataframe)
        
        if self.print_debug:
            print(dataframe)

        if isinstance(self.row_limit, int):
            if len(dataframe.index) > self.row_limit:
                if self.print_debug: print(f"Limiting number of rows: {len(dataframe.index)} -> {self.row_limit}")
                dataframe = dataframe.head(self.row_limit)

        
        self.header_texts = dataframe.columns.to_list()
        self.cells_values = dataframe.values.tolist()

        self.render_table()

    def destroy_table(self):
        destroy_all_children(self.parent_frame)
    
    def render_table(self):
        
        self.destroy_table()

        # headers
        header_labels = []
        for label_text in self.header_texts:
            header_labels.append(tk.Label(self.parent_frame, text=label_text, font=self.font_header, bg=memory.bg_color_label_in_table, fg=memory.bg_color_label_in_table_header))
        for i, _label in enumerate(header_labels):
            _label.grid(row=0, column=i)

        # create label widgets
        labels = []
        for row_values in self.cells_values:
            row_widgets = []
            for value in row_values:
                # if value is nan, convert to "-"
                try:
                    if np.isnan(value):
                        value = "-"
                except: pass

                if isinstance(value, (float, int, str)):
                    # round up numbers and convert to string with trailing zeros
                    if isinstance(self.round_values, int):
                        try:
                            value = f"%.{self.round_values}f" % round(value, self.round_values) # convert value to string and keep trailing zeros
                        except:
                            if self.print_debug: print(f"Value could not be rounded: {value}, type: {type(value)}")
                    if self.print_debug: print(f"Value kept: {value}, type: {type(value)}")
                    row_widgets.append(tk.Label(self.parent_frame, text=str(value), font=self.font_text, bg=memory.bg_color_label_in_table))
                else:
                    row_widgets.append(None)
                    if self.print_debug:
                        print(f"Value at discarded: {value}, type: {type(value)}")
            labels.append(row_widgets)
        

        # place widgets in grid
        for i_row, row_widgets in enumerate(labels):
            for i_column, widget in enumerate(row_widgets):
                if type(widget) == tk.Label: # skip None values
                    if i_column == 0 and self.first_col_allign_left: # first column aligned left
                        widget.grid(row=i_row+1, column=i_column, sticky="w")
                    else: # other columns aligned center
                        widget.grid(row=i_row+1, column=i_column)

def update_alfa_from_textbox(textfield_alfa) -> float:
    
    # update alfa value from textbox
    alfa_str_raw = textfield_alfa.get(1.0, tk.END)
    alfa_str_clean = alfa_str_raw.replace(" ", "")
    alfa_str_clean = alfa_str_clean.replace(",", ".")
    alfa_str_clean = alfa_str_clean.replace("\n", "")
    try:
        alfa_float = float(alfa_str_clean)
    except:
        error_message(f"Hodnota alfa není v číselném formátu. Pro oddělení desetinných čísel prosím použijte symbol tečky (.) a použijte pouze číslené znaky.\n\nVámi zadaná neplatná hodnota alfa:\n{alfa_str_raw} \n\nProgramem zpracovávaná hodnota:\n{alfa_str_clean}")
    
    if alfa_float < 0 or alfa_float > 1:
        error_message(f"Hodnota alfa musí být v intervalu (0, 1), avšak vámi zadaná hodnota je {alfa_float}")
    
    # replace alfa value in text field
    textfield_alfa.delete(1.0, tk.END)
    textfield_alfa.insert(1.0, str(alfa_str_clean))
    return alfa_float



def plot(figure: Figure, parent_window: tk.Frame, include_toolbar = True):
    """Visualizes matplotlib Figure onto selected tkinter Frame
    Args:
        figure (Figure): figure with plot
        parent_window (tk.Frame): parent frame to host the plot
        include_toolbar (bool): wether the plot shloud include toolbar with option like to export plot to image
    """

    frame = tk.Frame(parent_window)
    frame.pack(anchor="w")

	# creating the Tkinter canvas
	# containing the Matplotlib figure

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.draw()

	# placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack()

	# creating the Matplotlib toolbar 
    if include_toolbar:
        toolbar = NavigationToolbar2Tk(canvas, frame) 
        toolbar.update() 

	# placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack(anchor="w")



def boxplot(dataframe: pd.DataFrame, figure_size: tuple, dpi: int) -> Figure:
    """Returns figure with boxplot from pandas dataframe
    Args:
        dataframe (pd.DataFrame): data
        figure_size (tuple): tuple with figure size (width, height)
        dpi (int): dpi of the plot
    Returns:
        Figure: figure with boxplot
    """

    fig = Figure(figsize=figure_size, dpi=dpi)
    subplot = fig.add_subplot()

    dataframe.boxplot(ax=subplot)
    subplot.set_title("Krabičkový graf (boxplot)")
    subplot.set_xlabel("Skupiny")
    subplot.set_ylabel("Hodnoty")

    return fig