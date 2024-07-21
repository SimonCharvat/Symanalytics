

print("Načítání modulu: Homogenita")

from lib import memory
from lib import tools

import pandas as pd
import numpy as np
from scipy import stats


class Homogenity():
    def __init__(self, dataset: pd.DataFrame, selected_test: str) -> None:
        self.update_settings(dataset, selected_test)

    def update_settings(self, dataset: pd.DataFrame, selected_test: str):
        self.df = dataset
        self.selected_test = selected_test

        self.groups_count = len(self.df.columns) # count of group (coulumns) (k)
        self.counts = [] # count of values (not NaN) for every column
        self.means = [] # mean of values for every column
        self.medians = [] # median values for every column
        

        for col in self.df.columns:
            # calculate count (n)
            _n = self.df[col].count()
            self.counts.append(_n)
            
            # calculate mean
            _mean = self.df[col].sum(skipna=True) / _n
            self.means.append(_mean)

            # calculate median
            _median = np.median(self.df[col])
            self.medians.append(_median)
        
        self.n_total = sum(self.counts)
        self.total_mean = np.dot(self.means, self.counts) / self.n_total

        #print(self.groups_count)
        #print(self.counts)
        #print(self.means)
        #print(self.n_total)


    def calculate(self, alpha: float) -> dict:
        """
        Calculates selected homogenity test.
        """

        output = memory.get_empty_statistics_output_dict()

        if self.groups_count == 0 or len(self.df.index) == 0:
            output["runtime_result"] = "Test homogenity: nastala chyba"
            output["valid_test_bool"] = False
            return output
        
        if self.selected_test == "Bartlettův test":
            output = self.bartlett()
        elif self.selected_test == "Levenův test":
            output = self.leven()
        else:
            tools.error_message("Nebyl zvolen žádný test homogenity.")
            output["runtime_result"] = "Test homogenity: nastala chyba"
            output["valid_test_bool"] = False
            return output        
        
        output["test_used"] = self.selected_test
        output["alpha"] = alpha
        output["h0"] = "Rozptyly skupin jsou vzájemně shodné"
        output["h1"] = "Rozptyly skupin nejsou vzájemně shodné"
        output["date_time"] = tools.get_datetime_string()
        output["valid_output"] = True # test was successful

        if output["valid_test_bool"]: # if real test was selected
            if output["p_value"] > alpha:
                # no evidence against null hypothesis: p > alpha
                output["result"] = f"P-hodnota {round(output['p_value'], 3)} je větší než hodnota alfa {alpha}, což znamená, že není dostatek důkazů pro zamítnutí nulové hypotézy. Rozptyly jednotlivých skupin nejsou nijak významně odlišné."
                output["result_anova"] = "Předpoklad analýzy rozptylu je tedy splněn."
            else:
                # strong evidence against null hypothesis: p <= alpha
                output["result"] = f"P-hodnota {round(output['p_value'], 3)} je menší než hodnota alfa {alpha}, což znamená, že je dostatek důkazů pro zamítnutí nulové hypotézy ve prospěch alternativní. Rozptyl alespoň jedné skupiny je významně odlišný."
                output["result_anova"] = "Předpoklad analýzy rozptylu není splněn a nedoporučuje se provádět analýzu rozptylu. Její výsledky mohou být zavádějící."
        return output

    def leven(self):

        output = memory.get_empty_statistics_output_dict()
        output["runtime_result"] = "Test homogenity: proveden úspěšně"

        multiplier = (self.n_total - self.groups_count) / (self.groups_count - 1)

        numerator = []
        denominator = []
        z_i_values = []
        z_i_means = []
        
        # calculating 'z' values
        for i, col in enumerate(self.df.columns):
            zij_values = (self.df[col] - self.means[i]).abs()
            z_i_values.append(zij_values)
            z_i_means.append(np.nanmean(zij_values))
        
        z_ij_total_mean = np.dot(z_i_means, self.counts) / self.n_total

        for i, col in enumerate(self.df.columns):
            # between-group variability multiplied by the number of values in group (weight of the variablitiy)
            numerator.append(self.counts[i] * ((z_i_means[i] - z_ij_total_mean)**2))
            
            # within-group variability
            denominator.append(((z_i_values[i] - z_i_means[i])**2).sum(skipna=True))

        statistic = multiplier * sum(numerator) / sum(denominator)
        p_value = 1 - stats.f.cdf(statistic, self.groups_count - 1, self.n_total - self.groups_count)
        
        
        output["statistic"] = statistic
        output["p_value"] = p_value
        

        print("\nLevenův test homogenity:")
        print(f"Statistika: {statistic}")
        print(f"P-hodnota: {p_value}")

        return output


    def bartlett(self):
        output = memory.get_empty_statistics_output_dict()
        output["runtime_result"] = "Test homogenity: proveden úspěšně"

        # check minimum count
        error_count_text = "Bartlettův test homogenity nelze spočítat, protože alespoň jedna skupia má četnost menší než 3. Prosím využijte jiný test nebo doplňte dataset o další pozorování.\n\nProblémové skupiny:"
        error_count_exists = False
        for i, col in enumerate(self.df.columns):
            if self.counts[i] < 3:
                error_count_text = error_count_text + f"\nSkupina: {col}, Četnost: {self.counts[i]}"
                error_count_exists = True
        if error_count_exists:
            tools.error_message(error_count_text)
            output["runtime_result"] = "Test homogenity: nastala chyba"
            output["valid_test_bool"] = False
            return output
            

        numerator_sum = []
        denominator_sum = []
        pool_var_sum = []

        for i, col in enumerate(self.df.columns):
            numerator_sum.append(
                (self.counts[i] - 1) * np.log(self.df[col].var())
            )
            denominator_sum.append(
                1 / (self.counts[i] - 1)
            )
            pool_var_sum.append(
                (self.counts[i] - 1) * self.df[col].var()
            )

        s_var_p = 1 / (self.n_total - self.groups_count) * sum(pool_var_sum)
        
        numerator = (self.n_total - self.groups_count) * np.log(s_var_p) - sum(numerator_sum)
        denominator = 1 + 1 / (3 * (self.groups_count - 1)) * (sum(denominator_sum) - 1 / (self.n_total - self.groups_count))

        statistic = numerator / denominator
        p_value = 1 - stats.chi2.cdf(statistic, self.groups_count - 1)

        output["statistic"] = statistic
        output["p_value"] = p_value

        print("\nBartlettův test homogenity:")
        print(f"Statistika: {statistic}")
        print(f"P-hodnota: {p_value}")

        return output
    


    def no_test(self):
        print("Test homogenity nebyl proveden")
        output = memory.get_empty_statistics_output_dict()
        output["runtime_result"] = "Test homogenity: nebyl proveden"
        output["result"] = "Na základě zvolených parametrů nebyl test homogenity proveden"
        output["result_anova"] = "Test homogenity nebyl proveden a tedy nelze ověřit tento předpoklad analýzy rozptylu."


        return output



