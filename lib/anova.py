
print("Načítání modulu: Analýza rozptylu")

from lib import tools
from lib import memory
from lib import homogenity
from lib import normality
from lib import mct

import pandas as pd
import numpy as np
import tkinter as tk
import scipy.stats


class AnovaTest():

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.update_settings(dataset)

    def update_settings(self, dataset: pd.DataFrame):
        self.df = dataset

        self.groups_count = len(self.df.columns) # count of group (coulumns) (k)
        self.counts = [] # count of values (not NaN) for every column
        self.means = [] # mean of values for every column

        for col in self.df.columns:
            # calculate count (n)
            _n = self.df[col].count()
            self.counts.append(_n)
            
            # calculate mean
            _mean = self.df[col].sum(skipna=True) / _n
            self.means.append(_mean)
        
        self.n_total = sum(self.counts)
        self.total_mean = np.dot(self.means, self.counts) / self.n_total

        # check if all groups have the same number of (for selection of mct test)
        if len(set(self.counts)) == 1:
            self.equal_group_counts = True
        else:
            self.equal_group_counts = False


    def calculate(self, alpha: float, print_debug=False) -> dict:
        """
        Calculates anova.
        """
        output = {
            "h0": "Střední hodnoty skupin jsou shodné",
            "h1": "Střední hodnota alespoň jedné skupiny je odlišná",
            "alpha": alpha,
            "date_time": tools.get_datetime_string()
        }
        
        if self.groups_count == 0 or len(self.df.index) == 0:
            output["runtime_result"] = "Test ANOVA: nastala chyba"
            output["valid_test_bool"] = False
            return output
        
        output["valid_test_bool"] = True    
        output["valid_output"] = True
        output["runtime_result"] = "Test ANOVA: proveden úspěšně"

        df_anova = pd.DataFrame({
            "Skupina": self.df.columns.to_list(),
            "mean": self.means,
            "n": self.counts
        })

        # sum of squares - between groups
        _ss_between = np.dot((np.array(self.means) - self.total_mean) ** 2, self.counts) # sum{ n_i * (mean_i - mean_total)^2 }
        
        # sum of squares - within groups
        _ss_within_group_cumsum = 0
        for i, col in enumerate(self.df.columns):
            _ss_within_group_partial = sum((self.df[col][~self.df[col].isna()] - self.means[i]) ** 2)
            _ss_within_group_cumsum = _ss_within_group_cumsum + _ss_within_group_partial

        # sum of squares
        output["sum_of_squares"] = [
            _ss_between,
            _ss_within_group_cumsum,
            _ss_between + _ss_within_group_cumsum
        ]
        
        if print_debug:
            print("SS - between, within, total")
            print(output["sum_of_squares"])

        # degrees of freedom
        output["degrees_freedom"] = [
            self.groups_count - 1, # k - 1 = number of groups - 1 = between
            self.n_total - self.groups_count, # n - k = number of observations - number of groups = within
            self.n_total - 1 # n - 1 = number of observatios - 1 = total
        ]
        if print_debug:
            print("degrees_freedom")
            print(output["degrees_freedom"])

        # mean squares = SS / df
        output["mean_squares"] = [
            output["sum_of_squares"][0] / output["degrees_freedom"][0], # between
            output["sum_of_squares"][1] / output["degrees_freedom"][1] # within
        ]
        if print_debug:
            print("mean_squares")
            print(output["mean_squares"])

        # F-statistic = mean square error between groups / mean square error within groups
        output["statistic"] = output["mean_squares"][0] / output["mean_squares"][1]
        if print_debug:
            print("statistic")
            print(output["statistic"])

        # F-critical = invers of probability distribution function at specified level for specified degrees of freedom
        output["critical"] = scipy.stats.f.ppf(1-alpha, output["degrees_freedom"][0], output["degrees_freedom"][1])
        if print_debug:
            print("critical")
            print(output["critical"])

        # p-value = cumulative distribution function
        output["p_value"] = 1 - scipy.stats.f.cdf(output["statistic"], output["degrees_freedom"][0], output["degrees_freedom"][1])
        if print_debug:
            print("p_value")
            print(output["p_value"])

        if output["p_value"] > alpha:
            # no evidence against null hypothesis: p > alpha
            output["result"] = f"P-hodnota {round(output['p_value'], 3)} je větší než hodnota alfa {alpha}, což znamená, že není dostatek důkazů pro zamítnutí nulové hypotézy."
            output["result_anova"] = "Není dostatek důkazů pro zavrhnutí hypotézy, že všechny skupiny mají stejnou střední hodnotu"
            output["h0_result"] = True
        else:
            # strong evidence against null hypothesis: p <= alpha
            output["result"] = f"P-hodnota {round(output['p_value'], 3)} je menší než hodnota alfa {alpha}, což znamená, že je dostatek důkazů pro zamítnutí nulové hypotézy ve prospěch alternativní."
            output["result_anova"] = "Je dostatek důkazů pro zavrhnutí hypotézy o shodě středních hodnot. Lze předpokládat, že alespoň jedna dvojice skupin je vzájemně odlišná. Pro porovnání skupin je doporučeno využít post-hoc analýzu."
            output["h0_result"] = False
        return output



class AnovaInput():
    
    def __init__(self, frame_left, frame_right, root) -> None:

        self.frame_right = frame_right
        self.root = root
        
        # possible feature: give option to user to choose input format
        self.input_format = "long" # long or short

        self.df_anova = pd.DataFrame()

        # left panel button - data format
        tk.Button(frame_left, text="Nastavení analýzy", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")

        if self.root != None:
            tools.initial_resize_event(self.root)
        
    def update_df(self) -> bool:
        """
        Updates self.df_anova dataframe.
        Source dataframe is global memory.df variable.
        Based on selected numerical and categorical columns (column names saved in list memory.anova_data_input_col_names).
        New dataframe is saved as self.df_anova and is also return by this function.
        """
        col_values = memory.anova_data_input_col_names[0].get()
        col_categories = memory.anova_data_input_col_names[1].get()

        if not tools.check_variable(memory.df, col_values, require_number=True):
            return False
        if not tools.check_variable(memory.df, col_categories, require_number=False, unique_value_limit=8):
            return False
        
        match self.input_format:
            case "long":
                self.df_anova = tools.long_to_short(memory.df, col_categories, col_values)
            case "short": # possible feature: in GUI user must select which columns to include
                self.df_anova = memory.df
        
        # remove rows where all columns have nan value (source dataframe might be longer than seleced data)
        self.df_anova.dropna(how="all", inplace=True)

        print("Aktualizovaná tabulka pro analýzu rozptylu:")
        print(self.df_anova)
        return True

    def load_panel(self) -> None:
        
        tools.destroy_all_children(self.frame_right)

        # column names for data input [numerical, categorical]
        if memory.anova_data_input_col_names == None:
            memory.anova_data_input_col_names = [tk.StringVar(value="---"), tk.StringVar(value="---")]

        # initalize selected options
        if memory.anova_test_homogenity_var == None:
            memory.anova_test_homogenity_var = tk.StringVar(value=memory.homogenity_tests[0])
        if memory.anova_test_homogenity_var == None:
            memory.anova_test_mct_var = tk.StringVar(value=memory.mct_tests[0])

        # header
        tk.Label(self.frame_right, text="Jednofaktorová analýza rozptylu - nastavení", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        
        # data input - header, description
        tk.Label(self.frame_right, text="Vstupní data", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        text_1 = tk.Label(self.frame_right, font=memory.fonts["text"], bg=memory.bg_color_label, wraplength=300, justify="left", text="Pro provedení jednofaktorové analýzy rozptylu je za potřebí jedna numerické proměnná (udává hodnoty) a jedna katogoriální proměnná (rozděluje do skupin).")
        text_1.pack(anchor="w")
        self.frame_right.wrappable_labels.append(text_1)

        # data input - options
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        self.frame_data_input = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_data_input.pack(anchor="w")
        tk.Label(self.frame_data_input, text="Kategoriální proměnná:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        tk.Label(self.frame_data_input, text="Numerická proměnná:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=0, sticky="w")
        if len(memory.df.columns) > 0:
            tk.OptionMenu(self.frame_data_input, memory.anova_data_input_col_names[1], *memory.df.columns).grid(row=0, column=1, sticky="w")
            tk.OptionMenu(self.frame_data_input, memory.anova_data_input_col_names[0], *memory.df.columns).grid(row=1, column=1, sticky="w")
        else:
            tk.OptionMenu(self.frame_data_input, memory.anova_data_input_col_names[1], "---").grid(row=0, column=1, sticky="w")
            tk.OptionMenu(self.frame_data_input, memory.anova_data_input_col_names[0], "---").grid(row=1, column=1, sticky="w")
        tk.Label(self.frame_data_input, text="(rozdělení skupin)", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=2, sticky="w")
        tk.Label(self.frame_data_input, text="(hodnoty)", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=2, sticky="w")
            
        
        # selection of alpha value
        tk.Label(self.frame_data_input, text="Hodnota alfa:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=2, column=0, sticky="w")
        self.textfield_alfa = tk.Text(self.frame_data_input, height=1, width=7, font=memory.fonts["text"], bg=memory.bg_color_text)
        self.textfield_alfa.insert(tk.INSERT, memory.anova_homogenity["alpha"])
        self.textfield_alfa.grid(row=2, column=1, sticky="w")

        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # -- selection of statistical tests --
        
        # create variable
        if memory.anova_manual_testing == None:
            memory.anova_manual_testing = tk.BooleanVar(value=False)



        tk.Label(self.frame_right, text="Dodatečné nastavení:", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        
        self.frame_tests_selection_master_frame = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_tests_selection_master_frame.pack(anchor="w")
        
        tk.Label(self.frame_tests_selection_master_frame, text="Nastavit ručně:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(self.frame_tests_selection_master_frame, text="", onvalue=True, offvalue=False, variable=memory.anova_manual_testing, font=memory.fonts["text"], command=self.manual_test_selection_update, bg=memory.bg_color_checkbutton).grid(row=0, column=1, sticky="w")
        
        self.frame_tests_selection = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_tests_selection.pack(anchor="w")

        self.manual_test_selection_update()

        # empty row
        tk.Label(self.frame_right, text="", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")


        # tk.Button(self.frame_right, text="DeBug: Update anova dataframe", font=memory.fonts["text"], command=self.update_df).pack(anchor="w") obsolete because integrated into calculate()
        tk.Button(self.frame_right, text="Provést analýzu rozptylu", font=memory.fonts["text"], command=self.calculate, bg=memory.bg_color_button).pack(anchor="w")
        
        # -- runtime information about (partial) tests --
        # homogenity
        self.label_result_homogenity = tk.Label(self.frame_right, text="Test homogenity: Nebyl proveden", bg=memory.bg_color_label)
        self.label_result_homogenity.pack(anchor="w")
        # normality
        self.label_result_normality = tk.Label(self.frame_right, text="Test normality: Nebyl proveden", bg=memory.bg_color_label)
        self.label_result_normality.pack(anchor="w")
        # anova
        self.label_result_anova = tk.Label(self.frame_right, text="Analýza rozptylu: Nebyla provedena", bg=memory.bg_color_label)
        self.label_result_anova.pack(anchor="w")
        # mtc
        self.label_result_mtc = tk.Label(self.frame_right, text="Test mnohonásobného srovnání: Nebyl proveden", bg=memory.bg_color_label)
        self.label_result_mtc.pack(anchor="w")

        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)
    

    def manual_test_selection_update(self):

        
        if self.root != None:
            tools.initial_resize_event(self.frame_tests_selection)

        # declare initial variables
        if memory.anova_test_normality_bool == None:
            memory.anova_test_normality_bool = tk.BooleanVar(value=True)
        if memory.anova_test_homogenity_bool == None:
            memory.anova_test_homogenity_bool = tk.BooleanVar(value=True)
        if memory.anova_test_mct_bool == None:
            memory.anova_test_mct_bool = tk.BooleanVar(value=True)
        if memory.anova_test_homogenity_var == None:
            memory.anova_test_homogenity_var = tk.StringVar(value=memory.homogenity_tests[0])
        if memory.anova_test_mct_var == None:
            memory.anova_test_mct_var = tk.StringVar(value=memory.mct_tests[0])
        if memory.anova_qq_plot_bool == None:
            memory.anova_qq_plot_bool = tk.BooleanVar(value=True)
        if memory.anova_boxplot_bool == None:
            memory.anova_boxplot_bool = tk.BooleanVar(value=True)


        # hide test selection if manual testing is not selected
        tools.destroy_all_children(self.frame_tests_selection)
        if memory.anova_manual_testing == None:
            return
        if memory.anova_manual_testing.get() == False:
            return

        # empty row
        tk.Label(self.frame_tests_selection, text="", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        
        # normality test
        tk.Label(self.frame_tests_selection, text="Test normality", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=0, sticky="w")
        tk.Checkbutton(self.frame_tests_selection, onvalue=True, offvalue=False, variable=memory.anova_test_normality_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=1, column=1, sticky="w")
        self.label_normality_test = tk.Label(self.frame_tests_selection, text="Shapirův-Wilkův test", font=memory.fonts["text"], bg=memory.bg_color_label)
        self.label_normality_test.grid(row=1, column=2, sticky="w")

        # test of homogenity of variances
        tk.Label(self.frame_tests_selection, text="Test homogenity rozptylů", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=2, column=0, sticky="w")
        tk.Checkbutton(self.frame_tests_selection, onvalue=True, offvalue=False, variable=memory.anova_test_homogenity_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton, command=self.manual_test_selection_update).grid(row=2, column=1, sticky="w")
        self.dropdown_homogenity_test = tk.OptionMenu(self.frame_tests_selection, memory.anova_test_homogenity_var, *memory.homogenity_tests)
        self.dropdown_homogenity_test.grid(row=2, column=2, sticky="w")
        
        # test of mct
        tk.Label(self.frame_tests_selection, text="Test mnohonásobného srovnání", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=3, column=0, sticky="w")
        tk.Checkbutton(self.frame_tests_selection, onvalue=True, offvalue=False, variable=memory.anova_test_mct_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton, command=self.manual_test_selection_update).grid(row=3, column=1, sticky="w")
        self.dropdown_mct_test = tk.OptionMenu(self.frame_tests_selection, memory.anova_test_mct_var, *memory.mct_tests)
        self.dropdown_mct_test.grid(row=3, column=2, sticky="w")

        # qq plot
        tk.Label(self.frame_tests_selection, text="Q-Q graf", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=4, column=0, sticky="w")
        tk.Checkbutton(self.frame_tests_selection, onvalue=True, offvalue=False, variable=memory.anova_qq_plot_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=4, column=1, sticky="w")

        # qq plot
        tk.Label(self.frame_tests_selection, text="Boxplot", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=5, column=0, sticky="w")
        tk.Checkbutton(self.frame_tests_selection, onvalue=True, offvalue=False, variable=memory.anova_boxplot_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=5, column=1, sticky="w")


        # hide option menu if mct test is disabled
        if memory.anova_test_mct_bool.get(): self.dropdown_mct_test.grid()
        else: self.dropdown_mct_test.grid_forget()
        
        # hide option menu if homogenity test is disabled
        if memory.anova_test_homogenity_bool.get(): self.dropdown_homogenity_test.grid()
        else: self.dropdown_homogenity_test.grid_forget()

        # hide label if normality test is disabled
        if memory.anova_test_normality_bool.get(): self.label_normality_test.grid()
        else: self.label_normality_test.grid_forget()



    def calculate(self) -> None:
        
        # update alfa value from textbox
        alfa_float = tools.update_alfa_from_textbox(self.textfield_alfa)

        # updates self.df_anova dataframe based on selected columns
        if not self.update_df():
            return False # stops calculation if error occured

        # if not manual selection of tests: set normality test to True
        if not memory.anova_manual_testing.get():
            memory.anova_test_normality_bool.set(value=True)
        
        # calculate normality
        if memory.anova_test_normality_bool.get():
            self.normality = normality.Normality(self.df_anova, "Shapirův-Wilkův test")
            normality_output = self.normality.calculate(alfa_float)
            self.label_result_normality.config(text=normality_output["runtime_result"])
            if normality_output["valid_test_bool"] != False: memory.anova_normality = normality_output # normality sucessful
        else:
            memory.anova_normality = memory.get_empty_statistics_output_dict()
            self.label_result_normality.config(text="Test normality: Neprováděn")

        # if not manual selection of tests: select qq plot if normality breached
        if not memory.anova_manual_testing.get():
            if "recommend_qq_plot" in normality_output:
                if normality_output["recommend_qq_plot"]:
                    memory.anova_qq_plot_bool.set(value=True)
                else:
                    memory.anova_qq_plot_bool.set(value=False)
            else:
                # if error loading normality output: select qq plot
                memory.anova_qq_plot_bool.set(value=True)
        
        # generate QQ plot
        if memory.anova_qq_plot_bool.get():
            # if normality class instance was not created, create new with 'no test'
            if not memory.anova_test_normality_bool.get():
                self.normality = normality.Normality(self.df_anova, "no test")
            memory.anova_qq_plots_list = self.normality.qq_plots()
        else:
            memory.anova_qq_plots_list = None

        # if not manual selection of tests: select boxplot
        if not memory.anova_manual_testing.get():
            memory.anova_boxplot_bool.set(value=True)

        # generate boxplot
        if memory.anova_boxplot_bool.get():
            memory.anova_boxplot_figure = tools.boxplot(self.df_anova, memory.fig_size, memory.fig_dpi)
        else:
            memory.anova_boxplot_figure = None

        # if not manual selection of tests: select homogenity test based on normality result
        if not memory.anova_manual_testing.get():
            # if normality failed
            if not normality_output["valid_test_bool"] or "any_non_normal" not in normality_output:
                # throw error
                tools.error_message("Nastala chyba při testu normality, tudíž nebylo program nebyl schopný zvolit vhodný test homogenity. Prosím, vyřeště problém s testem normality nebo zvolte test homogenity manuálně nebo test homogenityzcela potlačte.")
                memory.anova_test_homogenity_bool.set(value=False)
            else:
                # continue - allow homogenity
                memory.anova_test_homogenity_bool.set(value=True)
            
                if normality_output["any_non_normal"]:
                    # at least one group has non-normal distribution -> levene test
                    print("Test homogenity: alespoň jedna skupina nemá normální rozdělení -> Levenův test")
                    memory.anova_test_homogenity_var.set(value="Levenův test")
                else:
                    # all groups have normal distribution -> bartlett test
                    print("Test homogenity: všechny skupiny mají normální rozdělení -> Bartlettův test")
                    memory.anova_test_homogenity_var.set(value="Bartlettův test")


        # calculate homogenity
        if memory.anova_test_homogenity_bool.get():
            self.homogenity = homogenity.Homogenity(self.df_anova, memory.anova_test_homogenity_var.get())
            homogenity_output = self.homogenity.calculate(alfa_float)
            self.label_result_homogenity.config(text=homogenity_output["runtime_result"])
            if homogenity_output["valid_test_bool"] != False: memory.anova_homogenity = homogenity_output # homogenity sucessful
        else:
            homogenity_output = memory.get_empty_statistics_output_dict()

        
        # calculate anova test
        self.anova = AnovaTest(self.df_anova)
        anova_output = self.anova.calculate(alfa_float)
        self.label_result_anova.config(text=anova_output["runtime_result"])
        if anova_output["valid_test_bool"] != False: memory.anova_test = anova_output # anova sucessful
        
        
        # if not manual selection of tests: set mct test by number of observations in groups
        if not memory.anova_manual_testing.get() and "h0_result" in anova_output:
            
            # if H0 (no difference): skip mct test, else (difference) do mct test
            if anova_output["h0_result"]:
                print("Všechny skupiny jsou shodné -> vynechána post-hoc analýza")
                memory.anova_test_mct_bool.set(value=False)
            else:
                print("Všechny skupiny nejsou shodné -> provedena post-hoc analýza")
                memory.anova_test_mct_bool.set(value=True)
            
                # if equal n for all groups: tukey, else scheffe
                if self.anova.equal_group_counts:
                    memory.anova_test_mct_var.set(value="Tukeyho test")
                    print("Stejný počet pozorování ve skupinách -> Tukeyho test")
                else:
                    memory.anova_test_mct_var.set(value="Scheffeho test")
                    print("Různý počet pozorování ve skupinách -> Scheffeho test")
            
        
        # calculate mct
        if memory.anova_test_mct_bool.get():
            self.mct = mct.mct(self.df_anova, memory.anova_test_mct_var.get())
            mtc_output = self.mct.calculate(alfa_float)
            self.label_result_mtc.config(text=mtc_output["runtime_result"])
            if mtc_output["valid_test_bool"] != False: memory.anova_mct = mtc_output # mct sucessful
        
        # if manual testing: reset tests
        if memory.anova_manual_testing.get():
            if not memory.anova_test_homogenity_bool.get():
                memory.anova_homogenity = memory.get_empty_statistics_output_dict()
            if not memory.anova_test_normality_bool.get():
                memory.anova_normality = memory.get_empty_statistics_output_dict()
            if not memory.anova_test_mct_bool.get():
                memory.anova_mct = memory.get_empty_statistics_output_dict()


class AnovaOutput():
    
    def __init__(self, frame_left, frame_right, root: tk.Tk) -> None:
        
        self.frame_right = frame_right
        self.root: tk.Tk = root

        # left panel button - data format
        tk.Button(frame_left, text="Výstup", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")

        if self.root != None:
            tools.initial_resize_event(self.root)
    
    
    def update_qq_plot_left(self): self.update_qq_plot_master(delta=-1)
    def update_qq_plot_right(self): self.update_qq_plot_master(delta=1)

    def update_qq_plot_master(self, delta: int):
        max_index = len(memory.anova_qq_plots_list) - 1

        # move index
        self.qq_plots_index = self.qq_plots_index + delta
        
        # if new index is invalid: rollback index and do nothing (code shloud never run - not possible to get to the state)
        if self.qq_plots_index > max_index or self.qq_plots_index < 0:
            self.qq_plots_index = self.qq_plots_index + delta
            return

        # if index == 0: disable left button, else enable left button
        if self.qq_plots_index == 0:
            self.button_qq_plots_left.configure(state="disabled")
        else:
            self.button_qq_plots_left.configure(state="normal")
        
        # if index == 0: disable right button, else enable right button
        if self.qq_plots_index == max_index:
            self.button_qq_plots_right.configure(state="disabled")
        else:
            self.button_qq_plots_right.configure(state="normal")

        self.label_qq_plots.configure(text=f"Zobrazena skupina {self.qq_plots_index + 1} z {max_index + 1}")

        tools.destroy_all_children(self.frame_qq_plots_frame)
        tools.plot(memory.anova_qq_plots_list[self.qq_plots_index], self.frame_qq_plots_frame, True)


    def load_panel(self) -> None:
    
        # rounding precision for tables
        precis = 3
        
        tools.destroy_all_children(self.frame_right)
        
        # header anova
        tk.Label(self.frame_right, text="Jednofaktorová analýza rozptylu - výstup", font=memory.fonts["header_huge"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # -- independece --
        self.frame_independence = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_independence.pack(anchor="w")
        tk.Label(self.frame_independence, text="Předpoklad: nezávislost pozorování", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
        label_independence = tk.Label(self.frame_independence, text="Test analýzy rozptylu předpokládá, že jsou jednotlivá pozorování ve vstupních datech na sobě nezávislá. Tento předpoklad lze nejlépe ověřit na základě znalosti metody sběru dat.", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        label_independence.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_independence)



        # -- normality --
        if memory.anova_normality["valid_output"]:
            tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            self.frame_normality = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
            self.frame_normality.pack(anchor="w")

            # normality - header, timedate
            tk.Label(self.frame_normality, text="Předpoklad: normální rozdělení", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_normality, text=f"{memory.anova_normality['test_used']}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_normality, text="Hypotéza:", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

            # normality - hypothetis H1, H0 - wrapable
            label_normality_h0 = tk.Label(self.frame_normality, text=f"H0: {memory.anova_normality['h0']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_normality_h1 = tk.Label(self.frame_normality, text=f"H1: {memory.anova_normality['h1']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_normality_h_text = tk.Label(self.frame_normality, text="Testy normality jsou pro kažou skupinu pozorování prováděny odděleně", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_normality_h0.pack(anchor="w")
            label_normality_h1.pack(anchor="w")
            label_normality_h_text.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_normality_h0)
            self.frame_right.wrappable_labels.append(label_normality_h1)
            self.frame_right.wrappable_labels.append(label_normality_h_text)

            # normality - output table of results
            if isinstance(memory.anova_normality["result_table"], pd.DataFrame):
                frame_normality_table = tk.Frame(self.frame_normality, bg=memory.bg_color_frame)
                frame_normality_table.pack(anchor="w")
                table_normality = tools.LabelsTable(frame_normality_table, memory.anova_normality["result_table"], memory.fonts["header"], memory.fonts["text"], precis, True)

                #table_normality = tools.DataTable(memory.anova_normality["result_table"], self.frame_normality, self.root)
                #table_normality.update_dataframe(memory.anova_normality["result_table"])

            # normality - text output interpretation of homogenity test - wrapable
            label_normality_result_partial = tk.Label(self.frame_normality, text=memory.anova_normality["result"], font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_normality_result_partial.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_normality_result_partial)
            
            # normality - text output interpretation of premise to anova - wrapable
            label_normality_result_anova = tk.Label(self.frame_normality, text=memory.anova_normality["result_anova"], font=memory.fonts["header"], bg=memory.bg_color_label, fg=memory.fg_color_label_highlight, justify="left")
            label_normality_result_anova.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_normality_result_anova)


        # qq plot
        if memory.anova_qq_plots_list != None:
            frame_qq_plots_general = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
            frame_qq_plots_general.pack(anchor="w")
            tk.Label(frame_qq_plots_general, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(frame_qq_plots_general, text="Q-Q graf", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(frame_qq_plots_general, text="Grafy slouží k vyhodnocení odchylky dat od normality", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            frame_qq_plots_grid = tk.Frame(frame_qq_plots_general, bg=memory.bg_color_frame)
            frame_qq_plots_grid.pack(anchor="w")
            self.button_qq_plots_left = tk.Button(frame_qq_plots_grid, text="Předchozí skupina", bg=memory.bg_color_button, font=memory.fonts["text"], command=self.update_qq_plot_left)
            self.button_qq_plots_right = tk.Button(frame_qq_plots_grid, text="Následující skupina", bg=memory.bg_color_button, font=memory.fonts["text"], command=self.update_qq_plot_right)
            self.button_qq_plots_left.grid(row=0, column=1, sticky="w")
            self.button_qq_plots_right.grid(row=0, column=2, sticky="w")
            self.label_qq_plots = tk.Label(frame_qq_plots_grid, text="--", font=memory.fonts["text"], bg=memory.bg_color_label)
            self.label_qq_plots.grid(row=0, column=3, sticky="w")

            # frame for plot itself
            self.frame_qq_plots_frame = tk.Frame(frame_qq_plots_general, bg=memory.bg_color_frame)
            self.frame_qq_plots_frame.pack(anchor="w")

            self.qq_plots_index = 0
            self.update_qq_plot_master(delta=0)


        # -- homogenity --
        if memory.anova_homogenity["valid_output"]:
            tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            self.frame_homogenity = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
            self.frame_homogenity.pack(anchor="w")

            # homogenity - header, timedate
            tk.Label(self.frame_homogenity, text="Předpoklad: homogenita rozptylu", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_homogenity, text=f"Použitý test: {memory.anova_homogenity['test_used']}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_homogenity, text="Hypotéza:", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")


            # homogenity - hypothetis H1, H0 - wrapable
            label_homogenity_h0 = tk.Label(self.frame_homogenity, text=f"H0: {memory.anova_homogenity['h0']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_homogenity_h1 = tk.Label(self.frame_homogenity, text=f"H1: {memory.anova_homogenity['h1']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_homogenity_h0.pack(anchor="w")
            label_homogenity_h1.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_homogenity_h0)
            self.frame_right.wrappable_labels.append(label_homogenity_h1)
            
            # homogenity - value outputs
            tk.Label(self.frame_homogenity, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_homogenity, text=f"Testová statistika: {round(memory.anova_homogenity['statistic'], 3)}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_homogenity, text=f"P-hodnota: {round(memory.anova_homogenity['p_value'], 3)}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            
            # homogenity - text output interpretation of homogenity test - wrapable
            label_homogeniny_result_partial = tk.Label(self.frame_homogenity, text=memory.anova_homogenity["result"], font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_homogeniny_result_partial.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_homogeniny_result_partial)

            # homogenity - text output interpretation of premise to anova - wrapable
            label_homogeniny_result_anova = tk.Label(self.frame_homogenity, text=memory.anova_homogenity["result_anova"], font=memory.fonts["header"], bg=memory.bg_color_label, justify="left", fg=memory.fg_color_label_highlight)
            label_homogeniny_result_anova.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_homogeniny_result_anova)


        # -- anova --
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        self.frame_anova = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_anova.pack(anchor="w")

        # anova - header, timedate
        tk.Label(self.frame_anova, text="Výsledek analýzy rozptylu (ANOVA)", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_anova, text="Hypotéza:", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # anova - hypothetis H1, H0 - wrapable
        label_anova_h0 = tk.Label(self.frame_anova, text=f"H0: {memory.anova_test['h0']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        label_anova_h1 = tk.Label(self.frame_anova, text=f"H1: {memory.anova_test['h1']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        label_anova_h0.pack(anchor="w")
        label_anova_h1.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_anova_h0)
        self.frame_right.wrappable_labels.append(label_anova_h1)
        
        # anova - value outputs
        tk.Label(self.frame_anova, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_anova, text=f"Testová statistika: {round(memory.anova_test['statistic'], 3)}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_anova, text=f"P-hodnota: {round(memory.anova_test['p_value'], 3)}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # anova - text output interpretation of premise to anova - wrapable
        label_anova_result_formal = tk.Label(self.frame_anova, text=memory.anova_test["result"], font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        label_anova_result_informal = tk.Label(self.frame_anova, text=memory.anova_test["result_anova"], font=memory.fonts["header"], bg=memory.bg_color_label, justify="left", fg=memory.fg_color_label_highlight)
        label_anova_result_formal.pack(anchor="w")
        label_anova_result_informal.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_anova_result_formal)
        self.frame_right.wrappable_labels.append(label_anova_result_informal)


        if memory.anova_test["valid_output"]:
            # anova - output table
            frame_anova_results = tk.Frame(self.frame_anova, bg=memory.bg_color_frame)
            frame_anova_results.pack(anchor="w")
            
            df_anova_result = pd.DataFrame({
                "Variabilita": ["Meziskupinová", "Vnitroskupinová", "Celkem"],
                "Suma čtverců": memory.anova_test["sum_of_squares"],
                "Stupně volnosti": memory.anova_test["degrees_freedom"],
                "Střední čtvercová chyba": [*memory.anova_test["mean_squares"], ""],
                "F-statistika": [memory.anova_test["statistic"], "", ""],
                "F-kritická": memory.anova_test["critical"],
                "P-hodnota": memory.anova_test["p_value"]
            })

            tools.LabelsTable(frame_anova_results, df_anova_result, memory.fonts["header"], memory.fonts["text"], precis, True)


        # -- mct --
        if memory.anova_mct["valid_output"]:
            tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            self.frame_mct = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
            self.frame_mct.pack(anchor="w")

            # mct - header, timedate
            tk.Label(self.frame_mct, text="Mnohonásobné porovnávání (post hoc)", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_mct, text="Hypotéza:", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

            # mct - hypothetis H1, H0 - wrapable
            label_mct_h0 = tk.Label(self.frame_mct, text=f"H0: {memory.anova_mct['h0']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_mct_h1 = tk.Label(self.frame_mct, text=f"H1: {memory.anova_mct['h1']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_mct_h0.pack(anchor="w")
            label_mct_h1.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_mct_h0)
            self.frame_right.wrappable_labels.append(label_mct_h1)
        
            tk.Label(self.frame_mct, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

            # mct - output table
            frame_mct_results = tk.Frame(self.frame_mct, bg=memory.bg_color_frame)
            frame_mct_results.pack(anchor="w")

        
            # mct - output table
            mct_dataframe = memory.anova_mct["results_dataframe"]
            table_mct = tools.LabelsTable(frame_mct_results, mct_dataframe, memory.fonts["header"], memory.fonts["text"], round_values=precis, first_col_allign_left=True)

            # mct - text output interpretation
            tk.Label(self.frame_mct, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            label_mct_result = tk.Label(self.frame_mct, text=memory.anova_mct["result"], font=memory.fonts["header"], fg=memory.fg_color_label_highlight, bg=memory.bg_color_label, justify="left")
            label_mct_result.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_mct_result)


        # boxplot
        if memory.anova_boxplot_figure != None:
            tools.plot(memory.anova_boxplot_figure, self.frame_right, True)


        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)
        #tools.initial_resize_event(self.frame_right)