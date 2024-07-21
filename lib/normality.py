
print("Načítání modulu: Normalita")

from lib import memory
from lib import tools

from scipy import stats
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkinter as tk
import statsmodels.api as sm


class Normality():
    def __init__(self, dataset: pd.DataFrame, selected_test: str) -> None:
        self.update_settings(dataset, selected_test)

    def update_settings(self, dataset: pd.DataFrame, selected_test: str):
        self.df = dataset
        self.selected_test = selected_test

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



    def calculate(self, alpha: float) -> dict:
        """
        Calculates selected normality test.
        """
        output = memory.get_empty_statistics_output_dict()

        if self.groups_count == 0 or len(self.df.index) == 0:
            output["runtime_result"] = "Test normality: nastala chyba"
            output["valid_test_bool"] = False
            return output
        
        if self.selected_test == "Shapirův-Wilkův test":
            output = self.shapiro()
        elif self.selected_test == "no test":
            output = memory.get_empty_statistics_output_dict()
        else:
            tools.error_message("Nebyl zvolen žádný test normality.")
            output["runtime_result"] = "Test normality: nastala chyba"
            output["valid_test_bool"] = False
            return output        
        
        output["test_used"] = self.selected_test
        output["alpha"] = alpha
        output["h0"] = "Náhodná pozorování pocházejí z normálního rozdělení"
        output["h1"] = "Náhodná pozorování nepocházejí z normálního rozdělení"
        output["date_time"] = tools.get_datetime_string()
        output["valid_output"] = True

        if output["valid_test_bool"]: # if real test was selected
            
            df_normality = pd.DataFrame({
                "Skupina": self.df.columns.to_list(),
                "Statistika": output["statistic"],
                "P-hodnota": output["p_value"],
                })
            
            # output["accurate"]
            output["accurate"] = np.array(output["accurate"], dtype=bool)

            h0_bools = np.array(df_normality["P-hodnota"] > alpha, dtype=bool)
            
            # add h0 to table to be user friendly
            df_normality["H0"] = ""
            df_normality.loc[h0_bools, "H0"] = "Nezamítnuta"
            df_normality.loc[~h0_bools, "H0"] = "Zamítnuta"


            #df_normality["H0"] = df_normality["P-hodnota"] > alpha

            result_texts = []
            
            # if at least one group violated requirement
            if (output["accurate"] == False).any():
                result_texts.append(f"Pro alespoň jednu skupinu byl porušen předpoklad testu normality ({output['requirement']}). Pro takové skupiny mohou být závěry provedené na základě p-honoty zavádějící. Zkontrolujte rozdělení také pomocí Q-Q grafu.")
                output["recommend_qq_plot"] = True

                # if any of the values with valid requirement denies normality H0
                if (h0_bools[output["accurate"]] == False).any():
                    result_texts.append(f"Nejméně jedna skupina pozorování, která splňuje předpoklady testu, zamítá nulovou hypotézu na hladině {alpha}. Pozorování pro danou skupinu nepochází z normálního rozdělení.")
                    output["result_anova"] = "Předpoklad normality je na základě testu porušen. Míru odchýlení od normality můžete také vyhohodnotit na základě Q-Q grafu. V případě lehkého porušení normality je možné provést analýzu rozptylu, avšak je nutné obezřetně interpretovat výsledky. V případě silného porušení není možné provést analýzu rozptylu. "
                
                # else if any of the values violating requirement denies normality H0
                elif (h0_bools[~output["accurate"]] == False).any():
                    result_texts.append(f"Skupiny splňující předpoklady nezamítají nulovou hypotézu. Avšak nejméně jedna skupina pozorování, která porušuje předpoklady testu, zamítá nulovou hypotézu na hladině {alpha}. Pozorování pro danou skupinu nepochází z normálního rozdělení. Tento závěr však nemusí být přesný, neboť byl porušen předpoklad ({output['requirement']}).")
                    output["result_anova"] = "Nebylo možné rozhodnout jednoznačně o normalitě dat, využijte k tomu prosím Q-Q graf."
                
                # else (no group denies normality H0)
                else:
                    result_texts.append(f"Na základě testu normality, porušující své předpoklady ({output['requirement']}), není dostatek důkazů pro zamítnutí nulové hypotézy ani u jedné ze skupin. Lze tedy předpokládat, že všechny pozorování pocházejí z patřičného normálního rozdělení.")
                    output["result_anova"] = "Předpoklad normality pro analýzu rozptylu je splněn (zvolený test normality však nemusí být vhodný a doporučuje se kontrola pomocí Q-Q grafu.)."

            # none of the groups violated requirement
            else:
                if (h0_bools == False).any():
                    result_texts.append(f"Nejméně jedna skupina pozorování zamítá nulovou hypotézu na hladině {alpha}. Pozorování pro danou skupinu nepochází z normálního rozdělení.")
                    output["result_anova"] = "Předpoklad normality je na základě testu porušen. Není doporučeno provádět analýzu rozptylu. Míru odchýlení od normality můžete také vyhohodnotit na základě Q-Q grafu."
                    output["recommend_qq_plot"] = True
                else:
                    result_texts.append("Není dostatek důkazů pro zamítnutí nulové hypotézy ani u jedné ze skupin. Lze tedy předpokládat, že všechny pozorování pocházejí z patřičného normálního rozdělení.")
                    output["result_anova"] = "Předpoklad normality není porušen, lze pokračovat v analýze rozptylu."
                    output["recommend_qq_plot"] = False
            
            output["result_table"] = df_normality
            
            print("\nTesto normality:")
            print(df_normality)

            # at least one group is non-normal (H0 denied)
            if (df_normality["H0"] == False).any():
                output["any_non_normal"] = True
            else:
                output["any_non_normal"] = False
            

            output["result"] = "\n".join(result_texts)

        return output

    def shapiro(self):
        """
        Shapiro-Wilk test works best for small samples.
        For larger samples, the statistc is calculated correctly, but p-value might be misleading.
        Documentation for 'stats.shapiro' recommends samples smaller than 5000 observations.
        """
       
        # prepared output message - will not be sent, if failed
        output = memory.get_empty_statistics_output_dict()
        output["runtime_result"] = "Test normality: proveden úspěšně"
       
        # requirement for this test
        _min_count = 3
        _max_count = 5000
        output["requirement"] = f"{_min_count} <= n <= {_max_count}"

        result_statistics = []
        result_p_values = []
        result_accurate_p = []
        

        for i, col in enumerate(self.df.columns):

            # get not valid values (not nan) - nans occure, because the data is split into separate columns
            values = self.df[col][self.df[col].notna()]
            #print(f"Column {i}: {col}")
            #print("Values:")
            #print(values)

            if len(values) >= _min_count:
                # shapiro test's p-value is inaccurate for n > 5000
                accurate = True if self.counts[i] <= _max_count else False
                statistic, p_value = stats.shapiro(values)
                result_accurate_p.append(accurate)
                result_statistics.append(statistic)
                result_p_values.append(p_value)
            else:
                result_accurate_p.append(False)
                result_statistics.append(np.NaN)
                result_p_values.append(np.NaN)
        
        output["statistic"] = result_statistics
        output["p_value"] = result_p_values
        output["accurate"] = result_accurate_p

        return output
    

    


    def no_test(self):
        print("Nebyl proveden test homogenity")
        output = memory.get_empty_statistics_output_dict()
        output["runtime_result"] = "Test homogenity: nebyl proveden"
        output["result"] = "Na základě zvolených parametrů nebyl test homogenity proveden"
        output["result_anova"] = "Test normality nebyl proveden a tedy nelze ověřit tento předpoklad analýzy rozptylu."


        return output



    def qq_plots(self) -> Figure:
        """Generates matplotlib plot (returned) which can be passed to plot function in tools.
        Args:
            parent_frame (tk.Frame): tkinter parent frame
        Returns:
            (list) Returns list of matplotlib figures with qq plots (one plot for each gourp/column)
        """

        list_of_figures = []

        for col in self.df.columns:
            
            # get values by column without nan
            values = self.df[col].dropna()

            #values = self.df.values.flatten() # get values from dataframe as 1D numpy array
            #values = values[~np.isnan(values)] # remove nan values

            fig_qq = Figure(figsize=memory.fig_size, dpi=memory.fig_dpi)
            subplot = fig_qq.add_subplot()
            sm.qqplot(values, line="45", fit=True, ax=subplot)

            subplot.set_title(f"Q-Q graf: {col}")
            subplot.set_xlabel("Teoretické kvantily")
            subplot.set_ylabel("Pozorované kvantily")

            list_of_figures.append(fig_qq)

        return list_of_figures