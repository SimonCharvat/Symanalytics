
# multiple comparison test
print("Načítání modulu: Porovnávání")

from lib import memory
from lib import tools

import pandas as pd
import tkinter as tk
import numpy as np
from scipy import stats



class mct():
    def __init__(self, dataset: pd.DataFrame, selected_test: str) -> None:
        self.update_settings(dataset, selected_test)

    def update_settings(self, dataset: pd.DataFrame, selected_test: str):
        self.df = dataset
        self.selected_test = selected_test

        self.groups_count = len(self.df.columns) # count of group (coulumns) (k)
        self.counts = [] # count of values (not NaN) for every column
        self.means = [] # mean of values for every column
        
        # column names for the output results table
        self.results_dataframe_columns = ["Dvojice skupin", "H0", "P-hodnota", "Rozdíl průměrů", "Dolní mez", "Horní mez"]

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
        Calculates selected multiple comparison test.
        """

        # in case test fails (othervise replaced by dictionary from test)
        output = memory.get_empty_statistics_output_dict()
        
        if self.groups_count == 0 or len(self.df.index) == 0:
            output["runtime_result"] = "Test mnohonásobného srovnání: nastala chyba"
            output["valid_test_bool"] = False
            return output
        
        
        if self.selected_test == "Tukeyho test":
            output = self.tukey(alpha)
        elif self.selected_test == "Scheffeho test":
            output = self.scheffe(alpha)
        else:
            tools.error_message("Nebyl zvolen žádný test normality.")
            output["runtime_result"] = "Test mnohonásobného srovnání: nastala chyba"
            output["valid_test_bool"] = False
            return output        
        
        print(f"{output['runtime_result']}")

        output["h0"] = "Střední hodnoty dané dvojice skupin jsou shodné"
        output["h1"] = "Střední hodnoty dané dvojice skupin nejsou shodné"
        output["test_used"] = self.selected_test
        output["alpha"] = alpha
        output["date_time"] = tools.get_datetime_string()
        output["valid_output"] = True

        if output["valid_test_bool"]: # if real test was selected
            
            df_mct: pd.DataFrame = output["results_dataframe"]

            result_texts = []
            
            # if at least one group is significantly different
            different_rows = df_mct["H0"] == "Zamítnuta"
            different_rows_any = different_rows.any()
            
            if different_rows_any:
                different_rows_count = sum(different_rows)
                result_texts.append(f"Alespoň v jednom případě došlo k zamítnutí nulové hypotézy o rovnosti průměru. Tedy byla zjištěna alespoň jedna ({different_rows_count}) dvojice skupin, která má významně odlišný průměr.")
                result_texts.append("Dvojice, které mají rozdíl průměrů významně rozdílný od nuly:")
                result_texts.append("\n".join(df_mct["Dvojice skupin"][different_rows].to_list()))

            # all groups have the same mean
            else:
                result_texts.append(f"V žádném případě nedošlo k zamítnutí nulové hypotézy o rovnosti průměru. Žádná skupina tedy nemá významně odlišný průměr vůči ostatním skupinám dle mnohonásobného srovnání ({self.selected_test}).")
            
            if "h0_result" in memory.anova_test:
                # all means are the same (by anova) == means are different (by mct)
                if memory.anova_test["h0_result"] == different_rows_any:
                    output["result_anova"] = "Závěry na základě analýzy rozptylzu (ANOVA) a mnohonásobného srovnání si odporují. Tato situace může nastat, neboť každý test je založen na jiné metodice."
            else:
                output["result_anova"] = ""
            

            output["result"] = "\n".join(result_texts)

            print("\nTest mnohonásobného porovnávání:")
            print(df_mct)
        
        return output




    def tukey(self, alpha: float):
    
        # prepared output message - will not be sent, if failed
        output = memory.get_empty_statistics_output_dict()
        results_dataframe = pd.DataFrame(columns=self.results_dataframe_columns)

        # requirement for this test
        output["requirement"] = f"n1 = n2 = ..."
        output["requirement_bool"] = True
        
        # error if different sizes of columns
        if set(self.counts).__len__() != 1:
            tools.error_message(
                f"Nebyl splněn předpoklad pro Tukeyho metodu. Počet pozorování musí být stejný pro každou skupinu ({output['requirement']}). Prosím, zvolte alternativní test mnohonásobného srovnání nebo upravte vstupní data.\n\nČetnosti:\n" +
                '\n'.join([f"Skupina: {var}, Četnost: {count}" for var, count in zip(self.df.columns.to_list(), self.counts)])
            )
            output["runtime_result"] = "Test mnohonásobného porovnávání: nastala chyba"
            output["valid_test_bool"] = False
            return output
        
        output["runtime_result"] = "Test mnohonásobného porovnávání: proveden úspěšně"
        output["valid_output"] = True

        n_total = self.n_total # count of all observations
        k = self.groups_count # count of groups
        mse = memory.anova_test["mean_squares"][1] # mean square error (withing groups)

        quantile = stats.studentized_range.ppf(1-alpha, k=k, df=n_total-k)
        multiplier = np.sqrt(mse / min(self.counts)) 
        interval_half_width = quantile * multiplier


        for i, col_i in enumerate(self.df.columns):
            for j, col_j in enumerate(self.df.columns):
                # do not compare the group twice (<) or with itself (=)
                if j <= i:
                    continue
                
                pair = f"({col_i}, {col_j})"
                difference = np.abs(self.means[i] - self.means[j])
                p_value = 1 - stats.studentized_range.cdf(difference / multiplier, k=k, df=n_total-k)
                h0 = "Zamítnuta" if p_value <= alpha else "Nezamítnuta"

                # ["Dvojice skupin", "H0", "P-hodnota", "Rozdíl průměrů", "Dolní mez", "Horní mez"]
                results_dataframe.loc[len(results_dataframe.index)] = [pair, h0, p_value, difference, difference-interval_half_width, difference+interval_half_width]

        output["results_dataframe"] = results_dataframe

        return output
    



    def scheffe(self, alpha: float):
    
        # prepared output message - will not be sent, if failed
        output = memory.get_empty_statistics_output_dict()
        results_dataframe = pd.DataFrame(columns=self.results_dataframe_columns)

        output["requirement_bool"] = True
        output["runtime_result"] = "Test mnohonásobného porovnávání: proveden úspěšně"
        output["valid_output"] = True

        n_total = self.n_total # count of all observations
        k = self.groups_count # count of groups
        mse = memory.anova_test["mean_squares"][1] # mean square error (withing groups)

        quantile = stats.f.ppf(1-alpha, dfn=k-1, dfd=n_total-k)

        for i, col_i in enumerate(self.df.columns):

            for j, col_j in enumerate(self.df.columns):

                # do not compare the group twice (<) or with itself (=)
                if j <= i:
                    continue
                
                pair = f"({col_i}, {col_j})"
                difference = np.abs(self.means[i] - self.means[j])
                interval_half_width = np.sqrt( (k-1) * mse * quantile * (1/self.counts[i] + 1/self.counts[j]))
                f_observed = difference ** 2 / (mse * (1/self.counts[i] + 1/self.counts[j]))
                p_value = 1 - stats.f.cdf(f_observed / (k-1), dfn=k-1, dfd=n_total-k)
                h0 = "Zamítnuta" if p_value <= alpha else "Nezamítnuta"

                # f_critical = (k-1) * quantile # f_critical might be wrong

                # ["Dvojice skupin", "H0", "P-hodnota", "Rozdíl průměrů", "Dolní mez", "Horní mez"]
                results_dataframe.loc[len(results_dataframe.index)] = [pair, h0, p_value, difference, difference-interval_half_width, difference+interval_half_width]


        output["results_dataframe"] = results_dataframe

        return output

    

    def no_test(self):
        print("Test mnohonásobného porovnávání nebyl proveden")
        output = memory.get_empty_statistics_output_dict()
        output["runtime_result"] = "Test mnohonásobného srovnání: nebyl proveden"
        output["result"] = "Na základě zvolených parametrů nebyl test mnohonásobného srovnání proveden"
        output["result_anova"] = "Nebyla provedena post hoc analýza"

        return output

