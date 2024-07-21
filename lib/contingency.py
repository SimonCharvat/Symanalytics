
print("Načítání modulu: Kontingenční tabulka")

from lib import tools
from lib import memory

import pandas as pd
import numpy as np
import tkinter as tk
import scipy.stats



class ContingencyTest():
    """Chi-squared test + coefficients (Cramer's V, Contingency coeficient)"""
    

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.update_settings(dataset)

    def update_settings(self, dataset: pd.DataFrame):
        self.df: pd.DataFrame = dataset


    def calculate(self, alpha: float, f1_col: str, f2_col: str, weight_col: str, weighted: bool) -> dict:

        output = {
            "h0": "Faktory jsou nezávislé",
            "h1": "Faktory jsou závislé",
            "alpha": alpha,
            "date_time": tools.get_datetime_string()
        }
        
        output["valid_test_bool"] = True    
        output["valid_output"] = True
        output["runtime_result"] = "Kontingenční analýza: provedena úspěšně"

        # if includes null value
        if self.df.isnull().values.any():
            output["valid_test_bool"] = False    
            output["valid_output"] = False
            output["runtime_result"] = "Kontingenční analýza: nastala chyba"
            tools.error_message("Vstupní data obsahují neplatnou (NaN) hodnotu, a proto nemohla být analýza provedena. Možnou příčinou jsou různě dlouhé sloupce faktorů.")
            return output

        f1_names = self.df[f1_col].unique().tolist()
        f2_names = self.df[f2_col].unique().tolist()

        # if crosstab is less then 2x2
        if len(f1_names) < 2 or len(f2_names) < 2:
            output["valid_test_bool"] = False    
            output["valid_output"] = False
            output["runtime_result"] = "Kontingenční analýza: nastala chyba"
            tools.error_message(f"Alanýza nebyla provedena. Každý zvolený faktor musí obsahovat alespoň dvě unikátní hodnoty (kategorie).\n\nFaktor 1: {len(f1_names)} unikátních hodnot\nFaktor 2: {len(f2_names)} unikátních hodnot")
            return output

        table_name = f"{f1_col} | {f2_col}"

        df_crosstab = pd.DataFrame(columns=[table_name] + f2_names)

        counts = []

        if not weighted:
            weight_col = "_count_internal_column"
            self.df[weight_col] = 1

        for f1_index, f1_value in enumerate(f1_names):
            factor_1_bool = self.df[f1_col] == f1_value
            counts_f2 = []
            
            for f2_index, f2_value in enumerate(f2_names):
                factor_2_bool = self.df[f2_col] == f2_value
                
                factors_both_bool = factor_1_bool & factor_2_bool

                count = self.df[weight_col][factors_both_bool].sum()
                counts_f2.append(count)
            
            counts.append(counts_f2)
            df_crosstab.loc[len(df_crosstab.index)] = [f1_value] + counts_f2

        # calculate marginal sums (by 1 factor)
        counts = np.array(counts)
        f1_sums = counts.sum(axis=1)
        f2_sums = counts.sum(axis=0)
        total_sum = sum(f1_sums)

        # save marginal sums to dataframe (for presentation of data)
        df_crosstab["Suma"] = f1_sums
        df_crosstab.loc[len(df_crosstab.index)] = ["Suma"] + list(f2_sums) + [total_sum]

        print("\nKontingenční tabulka (pozorované četnosti):")
        print(df_crosstab)

        #df_name_expected = "Očekávané četnosti"
        df_name_expected = table_name
        df_expected = pd.DataFrame(columns=[df_name_expected] + f2_names)
        df_expected[df_name_expected] = f1_names

        # calculate
        statistic_g = 0
        requirement_breached = False # expected value must be >= 5

        for i in range(len(f1_names)):
            for j in range(len(f2_names)):
                nominator = counts[i, j] - (f1_sums[i] * f2_sums[j]) / total_sum
                denominator = (f1_sums[i] * f2_sums[j]) / total_sum # expected value
                df_expected.iloc[i, j+1] = float(denominator) # save expected value (+1 for column because first column is column factor1 names)
                if denominator < 5:
                    requirement_breached = True
                statistic_g = statistic_g + (nominator ** 2) / denominator

        
        
        
        # calculate sums for expected values
        df_expected["Suma"] = df_expected.drop(columns=df_name_expected).sum(axis=1) # add sum column (ignore/drop first column with string name of factor)
        df_expected.loc[len(df_crosstab.index)] = ["Suma"] + df_expected.drop(columns=df_name_expected).sum(axis=0).to_list() # add sum row (ignore/drop first column with string name of factor and than add 'Suma' as row name)

        print("\nKontingenční tabulka (očekávané četnosti):")
        print(df_expected)

        

        # evaluate chi-suqared test
        deg_freedom = (len(f1_names) - 1) * (len(f2_names) - 1) # (n_rows - 1) * (n_columns - 1)
        critical = scipy.stats.chi2.ppf(1-alpha, df=deg_freedom)
        p_value = 1 - scipy.stats.chi2.cdf(statistic_g, deg_freedom)
        
        print("\nTest závislosti:")
        print(f"Chi-kvadrát statistika G: {statistic_g}")
        print(f"Kritická hodnota: {critical}")
        print(f"P-hodnota: {p_value}")

        if requirement_breached:
            requitements_text = "Porušen předpoklad: všechny očekávané četnosti musí být >= 5"
        else:
            requitements_text = "Předpoklad neporušen: všechny očekávané četnosti jsou >= 5"

        
        if p_value >= alpha:
            result = "P-hodnota je větší nebo rovna zvolené hladině alfa. Není dostatek důkazů pro zamítnutí nulové hypotézy. Nelze na základě testu usuzovat závoslost mezi faktory."
        else:
            result = "P-hodnota je menší než zvolená hladina alfa. Je dostatek důkazů pro zamítnutí nulové hypotézy. Lze tedy na základě testu usuzovat závislost mezi faktory. Pro zjištění míry závislosti můžete využít koeficienty uvedené níže."

        if requirement_breached:
            result = result + " Byl však porušen předpoklad. Uvedená interpretace tedy může být zavádějící. Pro korektní provedení testu získejte více pozorování nebo zvažte sloučení faktorů s malým počtem očekávaných pozorování."

        # contingency coefficient
        cont_coef = np.sqrt(statistic_g / (statistic_g + total_sum))
        print(f"Koeficient C: {cont_coef}")

        # cramer v
        min_dim = min(len(f1_names), len(f2_names))
        cramer_v = np.sqrt(statistic_g / ((min_dim-1) * total_sum))
        print(f"Cramerovo V: {cramer_v}")

        
        output["statistic"] = statistic_g
        output["requirement"] = requitements_text
        output["cont_coef"] = cont_coef
        output["cramer_v"] = cramer_v
        output["dataframe_observed"] = df_crosstab
        output["dataframe_expected"] = df_expected
        output["df"] = deg_freedom
        output["p_value"] = p_value
        output["critical"] = critical
        output["result"] = result
        output["valid_output"] = True

        return output



class ContingencyInput():
    
    def __init__(self, frame_left, frame_right, root) -> None:

        self.frame_right = frame_right
        self.root = root
        
        # initalize selected options
        #self.homogenity_test_var = tk.StringVar(value="---")
        #self.normality_test_var = tk.StringVar(value="---")
        #self.mct_test_var = tk.StringVar(value="---")

        self.df_contingency = pd.DataFrame()

        # left panel button - data format
        tk.Button(frame_left, text="Nastavení analýzy", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")

        if self.root != None:
            tools.initial_resize_event(self.root)
        
    def update_df(self) -> bool:
        """
        Updates self.df_contingency dataframe.
        Source dataframe is global memory.df variable.
        Based on selected columns.
        """
        col_factor_1 = memory.cont_data_input_col_name_factor_1.get()
        col_factor_2 = memory.cont_data_input_col_name_factor_2.get()
        col_weight = memory.cont_data_input_col_name_weight.get()

        include_weight_col = memory.contingency_include_weight_col.get()

        if not tools.check_variable(memory.df, col_factor_1, require_number=False, unique_value_limit=6):
            return False
        if not tools.check_variable(memory.df, col_factor_2, require_number=False, unique_value_limit=6):
            return False

        if include_weight_col:
            if not tools.check_variable(memory.df, col_weight, require_number=True):
                return False

        if include_weight_col:
            self.df_contingency = memory.df[[col_factor_1, col_factor_2, col_weight]]
        else:
            self.df_contingency = memory.df[[col_factor_1, col_factor_2]]

        # remove rows where all columns have nan value (source dataframe might be longer than seleced data)
        self.df_contingency.dropna(how="all", inplace=True)

        print("Aktualizovaná tabulka pro kontingenci:")
        print(self.df_contingency)
        return True



    def updated_weight_column_option(self) -> None:
        """
        Run this function when setting whether 'weight column is used' check box changes status.
        It disables dropdown menu selecting weight column and changes text next to the check box.
        """
        match memory.contingency_include_weight_col.get():
            case True:
                self.option_menu_weight_col.configure(state="active")
                self.check_button_weight_col.configure(text="Váženo přes tento sloupec")
            case False:
                self.option_menu_weight_col.configure(state="disabled")
                self.check_button_weight_col.configure(text="Neváženo")
            case _:
                tools.error_message("Nastala neočekávaná chyba. Prosím, restartujte program.")

    def load_panel(self) -> None:
        
        # column names for data input
        if memory.cont_data_input_col_name_factor_1 == None:
            memory.cont_data_input_col_name_factor_1 = tk.StringVar(value="---")
        if memory.cont_data_input_col_name_factor_2 == None:
            memory.cont_data_input_col_name_factor_2 = tk.StringVar(value="---")
        if memory.cont_data_input_col_name_weight == None:
            memory.cont_data_input_col_name_weight = tk.StringVar(value="---")
        
        tools.destroy_all_children(self.frame_right)
        
        # header
        tk.Label(self.frame_right, text="Kontigenční tabulka - nastavení", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        
        # data input - header, description
        tk.Label(self.frame_right, text="Vstupní data", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        text_1 = tk.Label(self.frame_right, font=memory.fonts["text"], bg=memory.bg_color_label, wraplength=300, justify="left", text="Pro provedení kontingenční analýzy je zapotřebí určit 2 proměnné jakožto faktory, podle kterých bude vytvořena kontingenční tabulka. V případě, že jeden řádek vstupních dat odpovídá jednomu pozorování, nezadávejte sloupec váhy. Jesltiže už máte vaše vstupní data agregovaná, je možné faktory vážit pomocí sloupce váha.")
        text_1.pack(anchor="w")
        self.frame_right.wrappable_labels.append(text_1)

        # data input - options
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        self.frame_data_input = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_data_input.pack(anchor="w")
        tk.Label(self.frame_data_input, text="Faktor 1:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        tk.Label(self.frame_data_input, text="Faktor 2:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=0, sticky="w")
        self.label_weight_col = tk.Label(self.frame_data_input, text="Váha:", font=memory.fonts["text"], bg=memory.bg_color_label)
        self.label_weight_col.grid(row=2, column=0, sticky="w")
        if memory.contingency_include_weight_col == None:
            memory.contingency_include_weight_col = tk.BooleanVar(None, False)
        self.check_button_weight_col = tk.Checkbutton(self.frame_data_input, text="", onvalue=True, offvalue=False, variable=memory.contingency_include_weight_col, font=memory.fonts["text"], command=self.updated_weight_column_option, bg=memory.bg_color_checkbutton)
        self.check_button_weight_col.grid(row=2, column=3, sticky="w")
        
        if len(memory.df.columns) > 0:
            tk.OptionMenu(self.frame_data_input, memory.cont_data_input_col_name_factor_1, *memory.df.columns).grid(row=0, column=1, sticky="w")
            tk.OptionMenu(self.frame_data_input, memory.cont_data_input_col_name_factor_2, *memory.df.columns).grid(row=1, column=1, sticky="w")
            self.option_menu_weight_col = tk.OptionMenu(self.frame_data_input, memory.cont_data_input_col_name_weight, *memory.df.columns)
            self.option_menu_weight_col.grid(row=2, column=1, sticky="w")
        else:
            tk.OptionMenu(self.frame_data_input, memory.cont_data_input_col_name_factor_1, "---").grid(row=0, column=1, sticky="w")
            tk.OptionMenu(self.frame_data_input, memory.cont_data_input_col_name_factor_2, "---").grid(row=1, column=1, sticky="w")
            self.option_menu_weight_col = tk.OptionMenu(self.frame_data_input, memory.cont_data_input_col_name_weight, "---")
            self.option_menu_weight_col.grid(row=2, column=1, sticky="w")
        

        self.updated_weight_column_option() # update to show initial text correctly for weight checkbox

        # selection of alpha value
        tk.Label(self.frame_data_input, text="Hodnota alfa:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=3, column=0, sticky="w")
        self.textfield_alfa = tk.Text(self.frame_data_input, height=1, width=7, font=memory.fonts["text"], bg=memory.bg_color_text)
        self.textfield_alfa.insert(tk.INSERT, memory.contingency_output["alpha"])
        self.textfield_alfa.grid(row=3, column=1, sticky="w")

        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")



        # tk.Button(self.frame_right, text="DeBug: Update anova dataframe", font=memory.fonts["text"], command=self.update_df).pack(anchor="w") obsolete because integrated into calculate()
        tk.Button(self.frame_right, text="Provést kontingenční analýzu", font=memory.fonts["text"], bg=memory.bg_color_button, command=self.calculate).pack(anchor="w")
        

        # runtime information - contingency
        self.label_result_contingency = tk.Label(self.frame_right, text="Kontingenční analýza: Nebyla provedena", bg=memory.bg_color_label, font=memory.fonts["text"])
        self.label_result_contingency.pack(anchor="w")


        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)
    
    
    def calculate(self) -> None:
        
        # update alfa value from textbox
        alfa_float = tools.update_alfa_from_textbox(self.textfield_alfa)
        
        # updates self.df_anova dataframe based on selected columns
        if not self.update_df():
            return False # stops calculation if error occured

        # calculate contingency
        contingency_test = ContingencyTest(self.df_contingency)
        contingency_output = contingency_test.calculate(
            alfa_float,
            memory.cont_data_input_col_name_factor_1.get(),
            memory.cont_data_input_col_name_factor_2.get(),
            memory.cont_data_input_col_name_weight.get(),
            memory.contingency_include_weight_col.get())
        self.label_result_contingency.config(text=contingency_output["runtime_result"])
        if contingency_output["valid_test_bool"] != False: memory.contingency_output = contingency_output # homogenity sucessful
        



class ContingencyOutput():
    
    def __init__(self, frame_left, frame_right, root: tk.Tk) -> None:
        
        self.frame_right = frame_right
        self.root: tk.Tk = root

        # left panel button - data format
        tk.Button(frame_left, text="Výstup", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")

        if self.root != None:
            tools.initial_resize_event(self.root)
    
    def load_panel(self) -> None:
    
        # rounding precision for tables
        precis = 3
        
        tools.destroy_all_children(self.frame_right)
        
        # header anova
        tk.Label(self.frame_right, text="Kontingenční tabulka - výstup", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")


        # -- independece --
        self.frame_independence = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_independence.pack(anchor="w")
        tk.Label(self.frame_independence, text="Předpoklad: nezávislost pozorování", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        label_independence = tk.Label(self.frame_independence, text="Test analýzy rozptylu předpokládá, že jsou jednotlivá pozorování ve vstupních datech na sobě nezávislá. Tento předpoklad lze nejlépe ověřit na základě znalosti metody sběru dat.", font=memory.fonts["text"], justify="left", bg=memory.bg_color_label)
        label_independence.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_independence)

        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # -- contingency --
        self.frame_contingency = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_contingency.pack(anchor="w")

        if memory.contingency_output["valid_output"]:
            
            # -- contingency counts tables --
           
            # chi-squared - observed counts table
            tk.Label(self.frame_contingency, text="Četnosti - pozorované", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
            frame_cont_results_observed = tk.Frame(self.frame_contingency, bg=memory.bg_color_frame)
            frame_cont_results_observed.pack(anchor="w")
            chisq_dataframe_observed = memory.contingency_output["dataframe_observed"]
            chisq_table_observed = tools.LabelsTable(frame_cont_results_observed, chisq_dataframe_observed, memory.fonts["header"], memory.fonts["text"], round_values=precis, print_debug=False)

            tk.Label(self.frame_contingency, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

            # chi-squared - expected counts table
            tk.Label(self.frame_contingency, text="Četnosti - očekávané", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
            frame_cont_results_expected = tk.Frame(self.frame_contingency, bg=memory.bg_color_frame)
            frame_cont_results_expected.pack(anchor="w")
            chisq_dataframe_expected = memory.contingency_output["dataframe_expected"]
            chisq_table_expected = tools.LabelsTable(frame_cont_results_expected, chisq_dataframe_expected, memory.fonts["header"], memory.fonts["text"], round_values=precis, print_debug=False)

            tk.Label(self.frame_contingency, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

            # chi-squared - header
            tk.Label(self.frame_contingency, text="Výsledek chí-kvadrát testu", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_contingency, text="Hypotéza:", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

            # chi-squared - hypothetis H1, H0 - wrapable
            label_anova_h0 = tk.Label(self.frame_contingency, text=f"H0: {memory.contingency_output['h0']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_anova_h1 = tk.Label(self.frame_contingency, text=f"H1: {memory.contingency_output['h1']}", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_anova_h0.pack(anchor="w")
            label_anova_h1.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_anova_h0)
            self.frame_right.wrappable_labels.append(label_anova_h1)

            # chi-squared - value outputs
            tk.Label(self.frame_contingency, text=f"Stupně volnosti: {round(memory.contingency_output['df'], precis)}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_contingency, text=f"Testová statistika: {round(memory.contingency_output['statistic'], precis)}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_contingency, text=f"Kritická hodnota: {round(memory.contingency_output['critical'], precis)}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_contingency, text=f"P-hodnota: {round(memory.contingency_output['p_value'], precis)}", font=memory.fonts["header"], bg=memory.bg_color_label, fg=memory.fg_color_label_highlight).pack(anchor="w")
            tk.Label(self.frame_contingency, text=memory.contingency_output["requirement"], font=memory.fonts["header"], bg=memory.bg_color_label, fg=memory.fg_color_label_highlight).pack(anchor="w")

            tk.Label(self.frame_contingency, text="", font=memory.fonts["text"]).pack(anchor="w")

            # chi-squared - text output interpretation - wrapable
            label_cont_result = tk.Label(self.frame_contingency, text=memory.contingency_output["result"], font=memory.fonts["header"], bg=memory.bg_color_label, justify="left", fg=memory.fg_color_label_highlight)
            label_cont_result.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_cont_result)

            tk.Label(self.frame_contingency, text="", font=memory.fonts["text"]).pack(anchor="w")



        # -- contingency coeficients --
        tk.Label(self.frame_contingency, text="Koeficienty", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        self.frame_cont_coef = tk.Frame(self.frame_contingency, bg=memory.bg_color_frame)
        self.frame_cont_coef.pack(anchor="w")
        if memory.contingency_output["valid_output"]:
            tk.Label(self.frame_cont_coef, text="Kontingenční koeficient", font=memory.fonts["header"], bg=memory.bg_color_label, fg=memory.fg_color_label_highlight).grid(row=0, column=0, sticky="w")
            tk.Label(self.frame_cont_coef, text="Cramerovo V:", font=memory.fonts["header"], bg=memory.bg_color_label, fg=memory.fg_color_label_highlight).grid(row=1, column=0, sticky="w")
            tk.Label(self.frame_cont_coef, text=round(memory.contingency_output["cont_coef"], precis), font=memory.fonts["header"], bg=memory.bg_color_label, fg=memory.fg_color_label_highlight).grid(row=0, column=1, sticky="w")
            tk.Label(self.frame_cont_coef, text=round(memory.contingency_output["cramer_v"], precis), font=memory.fonts["header"], bg=memory.bg_color_label, fg=memory.fg_color_label_highlight).grid(row=1, column=1, sticky="w")



        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)
        #tools.initial_resize_event(self.frame_right)