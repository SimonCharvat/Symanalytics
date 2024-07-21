
print("Načítání modulu: Korelace")

from lib import tools
from lib import correlation
from lib import memory

import pandas as pd
import numpy as np
import tkinter as tk
from scipy import stats
from matplotlib.figure import Figure




def pearson(mat: np.array, diagonal_value=1, print_output=True) -> np.array:
    """From given matrix, calculates pearson correlation matrix. Assumes that columns are variables and rows are observations.
    Used formula:
        correlation = suma{ x_i - mean_x } * suma{ y_i - mean_y } / (sqrt{suma{(x_i - mean_x)^2}} * sqrt{suma{(x_y - mean_y)^2}} )
    Args:
        mat (np.array): 2D array of input data
        diagonal_value (float): numerical value that will be on diagonal in output matrix
    Returns:
        np.array: 2D square correlation matrix
    """

    # gets dimentions of input matrix
    n_rows, n_cols = np.shape(mat)

    # calculates means of each column and saves as list (numpy array)
    means = np.mean(mat, axis=0)
    columns_centered = []

    # centers all variables (columns) by its means
    for i in range(n_cols):
        columns_centered.append(mat[:, i] - means[i])
    
    # creates empty square matrix
    r_mat = np.zeros((n_cols, n_cols))
    
    # fill diagonal with assigned value (mathematically is 1, but can se set as different)
    np.fill_diagonal(r_mat, diagonal_value)

    # calculates correlation matrix
    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                continue
            nominator = np.dot(columns_centered[i], columns_centered[j])
            denominator = np.sqrt(np.sum(columns_centered[i] ** 2)) * np.sqrt(np.sum(columns_centered[j] ** 2))
            if denominator == 0:
                r_mat[i, j] = np.nan
            else:
                r_mat[i, j] = nominator / denominator
    
    if print_output:
        print("\nPearsonův korelační koeficient:")
        print(r_mat)
    return r_mat


def spearman(mat: np.array, calculate_via_pearson=True) -> np.array:
    """From given matrix, calculates spearman correlation matrix. Assumes that columns are variables and rows are observations.
    Used method:
        Assignes rank to all values by columns and than parses it to pearson formula (spearman correlation matrix is pearson correllation applied to ranks). In case of same values in data, fractional ranks are used.
    Args:
        mat (np.array): 2D array of input data
        diagonal_value (float): numerical value that will be on diagonal in output matrix
    Returns:
        np.array: 2D square correlation matrix
    """

    # convert values to ranks by columns
    mat_rank = stats.rankdata(mat, axis=0)

    if calculate_via_pearson:
    
        r_mat_via_pearson = pearson(mat_rank, print_output=False)
        
        print("\nSpearmanův korelační koeficient:")
        print(r_mat_via_pearson)

        return r_mat_via_pearson


    n_rows, n_cols = np.shape(mat)
    r_mat_new = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            x = np.array(sorted(mat_rank[:, i]))
            y = np.array(sorted(mat_rank[:, j]))
            d = x - y
            r = 1 - (6 * np.sum(d ** 2)) / (n_rows * (n_rows**2 - 1))
            r_mat_new[i, j] = r
    
    print("\nSpearmanův korelační koeficient:")
    print(r_mat_new)

    return r_mat_via_pearson



class CorrelationInput():
    
    def __init__(self, frame_left, frame_right, root) -> None:

        # initalize selected options
        memory.corr_pearson_bool = tk.BooleanVar(value=True)
        memory.corr_spearman_bool = tk.BooleanVar(value=True)
        memory.corr_scatter_plot_bool = tk.BooleanVar(value=True)
        
        
        self.frame_right = frame_right
        self.root = root

        # left panel button - data format
        tk.Button(frame_left, text="Nastavení analýzy", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")

        if self.root != None:
            tools.initial_resize_event(self.root)

    
    def load_panel(self) -> None:
        
        tools.destroy_all_children(self.frame_right)
        
        # header
        tk.Label(self.frame_right, text="Korelační analýza - nastavení", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        
        # data input - header, description
        tk.Label(self.frame_right, text="Vstupní data", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        text_1 = tk.Label(self.frame_right, font=memory.fonts["text"], bg=memory.bg_color_label, wraplength=300, justify="left", text="Korelační analýza v programu zkoumá vzájemnou závislost proměnných po dvojicích. V případě korelačního koeficientu je výstupem korelační matice.")
        text_1.pack(anchor="w")
        self.frame_right.wrappable_labels.append(text_1)

        # data input - options
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Volba proměnných", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        self.frame_data_input = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_data_input.pack(anchor="w")

        if len(memory.df.columns) < 2:
            tk.Label(self.frame_right, text="Vstupní dataset musí obsahovat alespoň dvě proměnné (sloupece).", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_right, text="Nejprve prosím zvolte platný dataset.", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")

        # --- x variables ---
        # create list of boolean tkinter variables to know if column is in model
        if memory.corr_variable_bools == None:
            memory.corr_variable_bools = [tk.BooleanVar(value=False) for _ in range(len(memory.df.columns))]
        
        # create table of checkboxes for all columns
        self.vars_checkboxes = []
        self.vars_lables = []
        if len(memory.df.columns) > 0:
            for i, col in enumerate(memory.df.columns):
                self.vars_lables.append(
                    tk.Label(self.frame_data_input, text=col, font=memory.fonts["text"], bg=memory.bg_color_label)
                )
                self.vars_checkboxes.append(
                    tk.Checkbutton(self.frame_data_input, text="", onvalue=True, offvalue=False, variable=memory.corr_variable_bools[i], font=memory.fonts["text"], bg=memory.bg_color_checkbutton)
                )
                self.vars_lables[i].grid(row=i, column=0, sticky="w")
                self.vars_checkboxes[i].grid(row=i, column=1, sticky="w")


        # --- analysis setting ---
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Dodatečné nastaveníů", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        
        # grid frame for selection tests
        self.frame_settings = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_settings.pack(anchor="w")


        # pearson
        tk.Label(self.frame_settings, text="Pearsonův koralční koeficient", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(self.frame_settings, text="", onvalue=True, offvalue=False, variable=memory.corr_pearson_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=0, column=1, sticky="w")
        
        # spearman
        tk.Label(self.frame_settings, text="Spearmanův koralční koeficient", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=0, sticky="w")
        tk.Checkbutton(self.frame_settings, text="", onvalue=True, offvalue=False, variable=memory.corr_spearman_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=1, column=1, sticky="w")
        
        # scatter plot
        tk.Label(self.frame_settings, text="Bodový graf", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=2, column=0, sticky="w")
        tk.Checkbutton(self.frame_settings, text="", onvalue=True, offvalue=False, variable=memory.corr_scatter_plot_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=2, column=1, sticky="w")
        

        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        tk.Button(self.frame_right, text="Provést korelační analýzu", font=memory.fonts["text"], bg=memory.bg_color_button, command=self.calculate).pack(anchor="w")
        
        # runtime information - correlation
        self.label_result_correlation = tk.Label(self.frame_right, text="Korelační analýza: Nebyla provedena", bg=memory.bg_color_label)
        self.label_result_correlation.pack(anchor="w")

        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)
    


    def calculate(self) -> None:

        output = memory.get_empty_statistics_output_dict()

        selected_columns = []
        for i, col_bool in enumerate(memory.corr_variable_bools):
            # if column shloud be included
            if col_bool.get():
                # get column name
                col_name = memory.df.columns[i]
                
                if not tools.check_variable(memory.df, col_name, require_number=True):
                    output["runtime_result"] = "Korelační analýza: nastala chyba (neplatná proměnná)"
                    output["valid_test_bool"] = False
                    return output


                # create list of selected columns
                selected_columns.append(col_name)
        
        if len(selected_columns) < 2:
            tools.error_message("Pro korelační analýzu je nutné zvolit nejméně 2 proměnné")
            output["runtime_result"] = "Korelační analýza: nastala chyba"
            output["valid_test_bool"] = False
            return output

        # dataframe used to clean data
        df_correlation = memory.df[selected_columns].copy()

        # handle NaN values
        df_correlation.dropna(how="all", inplace=True)
        if df_correlation.isnull().values.any():
            tools.error_message("Nelze provést lineární regresi, protože některý ze zvolených sloupců obsahuje neplatnou (NaN) hodnotu")
            output["runtime_result"] = "Korelační analýza: nastala chyba"
            output["valid_test_bool"] = False
            return output

        df_as_matrix = df_correlation.values

        # pearson
        if memory.corr_pearson_bool.get():
            corr_mat_pearson = correlation.pearson(df_as_matrix)
            corr_df_pearson = pd.DataFrame(corr_mat_pearson, columns=selected_columns) # create dataframe and add names to columns
            corr_df_pearson.insert(0, column="-", value=selected_columns) # add names to rows
            output["pearson"] = corr_df_pearson
        else:
            output["pearson"] = None

        # spearman
        if memory.corr_spearman_bool.get():
            corr_mat_spearman = correlation.spearman(df_as_matrix)
            corr_df_spearman = pd.DataFrame(corr_mat_spearman, columns=selected_columns) # create dataframe and add names to columns
            corr_df_spearman.insert(0, column="-", value=selected_columns) # add names to rows
            output["spearman"] = corr_df_spearman
        else:
            output["spearman"] = None
        
        # scatter plots
        if memory.corr_spearman_bool.get():
            
            list_of_plots = []
            
            # generate scatter plot for each combinantion columns
            for col_idx_1, col_name_1 in enumerate(df_correlation.columns):
                for col_idx_2, col_name_2 in enumerate(df_correlation.columns):

                    # skip (if comparing column with it self)[=] OR (if already been compared)[<]
                    if col_idx_2 <= col_idx_1:
                        continue

                    fig = Figure(figsize=memory.fig_size, dpi=memory.fig_dpi)
                    subplot = fig.add_subplot()

                    subplot.scatter(df_correlation[col_name_1], df_correlation[col_name_2])

                    subplot.set_title(f'Proměnná "{col_name_2}" vůči proměnné "{col_name_1}"')
                    subplot.set_xlabel(col_name_1)
                    subplot.set_ylabel(col_name_2)

                    list_of_plots.append(fig)

            output["scatter_plot_list"] = list_of_plots
        else:
            output["scatter_plot_list"] = None


        
        output["valid_test_bool"] = True    
        output["valid_output"] = True
        output["runtime_result"] = "Korelační analýza: provedena úspěšně"

        memory.corr_output = output
        self.label_result_correlation.configure(text=output["runtime_result"])



class CorrelationOutput():
    
    def __init__(self, frame_left, frame_right, root: tk.Tk) -> None:
        
        self.frame_right = frame_right
        self.root: tk.Tk = root

        # left panel button - data format
        tk.Button(frame_left, text="Výstup", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")

        if self.root != None:
            tools.initial_resize_event(self.root)
    

    def update_plot_left(self): self.update_plot_master(delta=-1)
    def update_plot_right(self): self.update_plot_master(delta=1)

    def update_plot_master(self, delta: int):
        max_index = len(memory.corr_output["scatter_plot_list"]) - 1

        # move index
        self.plots_index = self.plots_index + delta
        
        # if new index is invalid: rollback index and do nothing (code shloud never run - not possible to get to the state)
        if self.plots_index > max_index or self.plots_index < 0:
            self.plots_index = self.plots_index + delta
            return

        # if index == 0: disable left button, else enable left button
        if self.plots_index == 0:
            self.button_plots_left.configure(state="disabled")
        else:
            self.button_plots_left.configure(state="normal")
        
        # if index == 0: disable right button, else enable right button
        if self.plots_index == max_index:
            self.button_plots_right.configure(state="disabled")
        else:
            self.button_plots_right.configure(state="normal")

        self.label_plots.configure(text=f"Zobrazen graf {self.plots_index + 1} z {max_index + 1}")

        tools.destroy_all_children(self.frame_plots_frame)
        tools.plot(memory.corr_output["scatter_plot_list"][self.plots_index], self.frame_plots_frame, True)


    def load_panel(self) -> None:
    
        # rounding precision for tables
        precis = 3
        
        tools.destroy_all_children(self.frame_right)
        
        # header
        tk.Label(self.frame_right, text="Korelační analýza - výstup", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")


        # pearson correlation matrix
        if memory.corr_pearson_bool.get() and "pearson" in memory.corr_output:
            if "pearson" in memory.corr_output:
                if isinstance(memory.corr_output["pearson"], pd.DataFrame):
                    tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                    tk.Label(self.frame_right, text="Pearsonův korelační koeficient", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
                    frame_pearson_table = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
                    frame_pearson_table.pack(anchor="w")
                    tools.LabelsTable(frame_pearson_table, memory.corr_output["pearson"], memory.fonts["header"], memory.fonts["text"], round_values=precis, first_col_allign_left=True)

        # spearman correlation matrix
        if memory.corr_spearman_bool.get():
            if "spearman" in memory.corr_output:
                if isinstance(memory.corr_output["spearman"], pd.DataFrame):
                    tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                    tk.Label(self.frame_right, text="Spearmanův korelační koeficient", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
                    frame_spearman_table = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
                    frame_spearman_table.pack(anchor="w")
                    tools.LabelsTable(frame_spearman_table, memory.corr_output["spearman"], memory.fonts["header"], memory.fonts["text"], round_values=precis, first_col_allign_left=True)
        
        # scatter plot
        if memory.corr_scatter_plot_bool.get():
            if "scatter_plot_list" in memory.corr_output:
                if isinstance(memory.corr_output["scatter_plot_list"], list):
                    
                    frame_plots_general = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
                    frame_plots_general.pack(anchor="w")
                    tk.Label(frame_plots_general, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                    tk.Label(frame_plots_general, text="Bodové grafy dle proměnných", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
                    tk.Label(frame_plots_general, text="Grafy slouží ke grafickému posouzení vztahu proměnných ", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                    frame_plots_grid = tk.Frame(frame_plots_general, bg=memory.bg_color_frame)
                    frame_plots_grid.pack(anchor="w")
                    self.button_plots_left = tk.Button(frame_plots_grid, text="Předchozí skupina", bg=memory.bg_color_button, font=memory.fonts["text"], command=self.update_plot_left)
                    self.button_plots_right = tk.Button(frame_plots_grid, text="Následující skupina", bg=memory.bg_color_button, font=memory.fonts["text"], command=self.update_plot_right)
                    self.button_plots_left.grid(row=0, column=1, sticky="w")
                    self.button_plots_right.grid(row=0, column=2, sticky="w")
                    self.label_plots = tk.Label(frame_plots_grid, text="--", font=memory.fonts["text"], bg=memory.bg_color_label)
                    self.label_plots.grid(row=0, column=3, sticky="w")

                    # frame for plot itself
                    self.frame_plots_frame = tk.Frame(frame_plots_general, bg=memory.bg_color_frame)
                    self.frame_plots_frame.pack(anchor="w")

                    self.plots_index = 0
                    self.update_plot_master(delta=0)


        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)
        #tools.initial_resize_event(self.frame_right)