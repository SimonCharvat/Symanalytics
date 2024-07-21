
print("Načítání modulu: Lineární regrese")


from lib import tools
from lib import memory
from lib import correlation
from lib import normality

import pandas as pd
import numpy as np
import tkinter as tk
from scipy import stats
from matplotlib.figure import Figure



class RegressionLinear():

    def fit_model(self, x: np.array, y: np.array, has_intercept: bool, print_output: bool = False):
        """Calculates beta coeffitients / fits line to data
        Args:
            x (np.array): 2D matrix of independent variables
            y (np.array): 1D array of dependent variable
            has_intercept (bool): if X matrix has intercept or not
            print_output (bool): wether output should be printed to console (default False)
        Returns:
            bool: wehther the matrix (X^T * X) is singular or not (if singular, error occured)
            np.array: beta coeffitients for independent variables
            np.array: predicted dependent values based on calculated beta coeffitients
            np.array: array of residuals (observed value - predicted value)
            float: coeffitient of determinantnion R^2
        """

        # (X^T * X)
        partial_mat = np.matmul(x.T, x)
        
        determinant = np.linalg.det(partial_mat)
        if determinant < 0.0001:
            error_singular = True
            print("Chyba při výpočtu: matice (X^T * X) je singulární")
            return error_singular, None, None, None, None
        else:
            error_singular = False

        
        # ordinary least square beta coefficients = (X^T * X)^-1 * X^T * y
        beta_coefficients = np.matmul(
            np.matmul(
                np.linalg.inv(partial_mat), x.T
            ),
            y
        )

        # predict y values
        y_pred = np.dot(x, beta_coefficients)

        # sum os quares of variable = variance of y
        y_ss = np.sum((y - np.mean(y)) ** 2)

        # array of residuals
        residuals = (y - y_pred)
        
        # sum of squares of residuals
        residual_ss = np.sum(residuals ** 2)

        if has_intercept:
            # r_squared = coeffitient of determinantion = sum of squares / residual sum of square
            if y_ss == 0:
                r_squared = np.nan
            else:
                r_squared = 1 - (residual_ss / y_ss)
        else:
            y_squared_sum = sum(y ** 2)
            if y_squared_sum == 0:
                r_squared = np.nan
            else:
                r_squared = 1 - (residual_ss / y_squared_sum)

        if print_output:
            print("\nModel lineární regrese:")
            print(f"Beta koeficienty: {beta_coefficients}")
            print(f"R^2: {r_squared}")

        return error_singular, beta_coefficients, y_pred, residuals, r_squared




    def perform_t_tests(self, x: np.array, beta: np.array, residuals: np.array):
        """Calculates t-tests for each regression coeffitient.
        Args:
            x (np.array): 2D matrix of independent variables
            beta (np.array): estimated beta coeffitients for independent variables
            residuals (np.array): array of residuals
        Returns:
            np.array: standard errors of regression coeffitients beta
            np.array: t-statistic of regression coeffitients beta
            np.array: p-values of regression coeffitients beta
        """

        n = x.shape[0] # number of observations (rows)
        p = x.shape[1] # number of predictors including intercept

        # degrees of freedom ... loosing 'p' degrees of freedom, because it is the number of unknown (only estimated) coeffitients
        degrees_of_freedom = n - p
        
        residual_ss = np.sum((residuals - np.mean(residuals)) ** 2)
        
        # sigma^2 = variance of error ... in this case we estimate sigma^2 by formula SSR / degrees of freedom
        sigma_2 = residual_ss / degrees_of_freedom

        # covariance matrix of beta estimates = sigma^2 * (X^T * X)^-1 ... where sigma^2 is estimated
        covariance_matrix = sigma_2 * np.linalg.inv(np.matmul(x.T, x))

        # standard errors of regression coeffitients beta = sqrt(diagonal elements of covariance matrix)
        standard_errors = np.sqrt(np.diagonal(covariance_matrix))

        # t-statistics
        t_statistics = beta / standard_errors

        # p-value = 2 * (1 - cdf(t_statistic)) ... multuiplied by 2 because it is two-sided test
        p_values = 2 * stats.t.sf(np.abs(t_statistics), degrees_of_freedom)

        print("\nT-test:")
        print(f"Standard errors: {standard_errors}")
        print(f"T-statistika: {t_statistics}")
        print(f"P-hodnota: {p_values}")

        return standard_errors, t_statistics, p_values


    def perform_breusch_pagan_test(self, x: np.array, residuals: np.array, alpha: float, deg_freedom: int, has_intercept: bool):
        """Performs Breusch-Pagan heteroskedasticity test. The test is performed by fitting new regression model.

        Args:
            x (np.array): 2D array of independent variables including intercept
            residuals (np.array): 1D array of residuals from original regression
            alpha (float): alpha value
            deg_freedom (int): degrees of freedom = number of parameters excluding intercept
            has_intercept (bool): if X matrix has intercept or not
        Returns:
            float: chi_squared statistic of heteroskedasticity test
            float: critical value of heteroskedasticity test
            float: p-values of heteroskedasticity test
            str: text interpretation of the test result
        """

        n = x.shape[0] # number of observations (rows)
        p = x.shape[1] # number of predictors including intercept
        
        residuals_squared = residuals ** 2 # square the residuals from original regression model

        # fit data to new model where dependent variables are squared residuals of original model (we extract R^2 from the new mode)
        error_singular, _, _, _, r_squared_resid = self.fit_model(x, residuals_squared, has_intercept=has_intercept) # 'has intercept' does only effect R^2 wchih is not even used by this test
        if error_singular:
            return np.nan, np.nan, np.nan, "Nastala chyba při výpočtu"

        # chi_squared statistic = new R^2 * number of observations
        statistic = r_squared_resid * n

        critical = stats.chi2.ppf(1-alpha, df=deg_freedom)
        p_value = 1 - stats.chi2.cdf(statistic, deg_freedom)

        if p_value < alpha:
            result_text = "Byla zamítnuta nulová hypotéza, což indikuje, že rozptyl residuí není konstantní. Je tedy porušen předpoklad lineární regrese. Odhady regresních parametrů nejsou nejlepším odhadem, zvažte použití jiného modelu."
        else:
            result_text = "Nebyla zamítnuta nulová hypotéza, což indikuje, že rozptyl residuí je konstantní. Není porušen předpoklad lineární regrese."

        print("\nBreusch-Paganův test heteroskedasticity")
        print(f"Chi-kvadrát statistika: {statistic}")
        print(f"Kritická hodnota: {critical}")
        print(f"P-hodnota: {p_value}")

        return statistic, critical, p_value, result_text



class RegressionInput():
    
    def __init__(self, frame_left, frame_right, root) -> None:

        self.frame_right = frame_right
        self.root = root

        # left panel button - data format
        tk.Button(frame_left, text="Nastavení analýzy", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")

        if self.root != None:
            tools.initial_resize_event(self.root)


    def disable_independent_variable_from_dependent(self, value=None):
        """
        Run this function when dependent variable selection changes (OptionMenu widget updates),
        Disables independent variable checkbox for variable already selected as independent.
        Value argument should be ignored, it is never used. Only kept, because tkinter calls functions with 1 positional argument.
        """
        
        if memory.reg_y == None:
            return
        
        independent_col = memory.reg_y.get()
        
        if independent_col == "---" or independent_col == None:
            return
        
        col_position = memory.df.columns.get_loc(independent_col)

        for i in range(len(self.x_lables)):
            if i == col_position:
                self.x_lables[i].configure(font=memory.fonts["text_disabled"])
                self.x_checkboxes[i].configure(state="disabled", text="Vysvětlovaná proměnná")
            else:
                self.x_lables[i].configure(font=memory.fonts["text"])
                self.x_checkboxes[i].configure(state="active", text="")

    
    
    def load_panel(self) -> None:
        
        tools.destroy_all_children(self.frame_right)
        
        # initalize selected options
        if memory.reg_y == None: memory.reg_y = tk.StringVar(value="---")
        if memory.reg_intercept == None: memory.reg_intercept = tk.BooleanVar(value=True)
        
        # header
        tk.Label(self.frame_right, text="Lineární regrese - nastavení", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        
        # data input - header, description
        tk.Label(self.frame_right, text="Vstupní data", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        text_1 = tk.Label(self.frame_right, font=memory.fonts["text"], bg=memory.bg_color_label, wraplength=300, justify="left", text="Pro lineární regresy je zapotřebí zvolit numerickou proměnnou, která bude vysvětlovaná, a alespoň jednu numerickou proměnnou, která bude vysvětlující. Dále je možnost zvolit, jeslti má regresní model obsahovat intercept či nikoliv.")
        text_1.pack(anchor="w")
        self.frame_right.wrappable_labels.append(text_1)

        # data input - options
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Volba vstupních proměnných", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        self.frame_data_input = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_data_input.pack(anchor="w")

        if len(memory.df.columns) < 2:
            tk.Label(self.frame_right, text="Vstupní dataset musí obsahovat alespoň dvě proměnné (sloupece).", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_right, text="Nejprve prosím zvolte platný dataset.", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")

        # y variable
        tk.Label(self.frame_data_input, text="Vysvětlovaná proměnná (y):", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        if len(memory.df.columns) > 0:
            tk.OptionMenu(self.frame_data_input, memory.reg_y, *memory.df.columns, command=self.disable_independent_variable_from_dependent).grid(row=0, column=1, sticky="w")
        else:
            tk.OptionMenu(self.frame_data_input, memory.reg_y, "---").grid(row=0, column=1, sticky="w")

        # intercept
        tk.Label(self.frame_data_input, text="Přidat intercept:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=0, sticky="w")
        tk.Checkbutton(self.frame_data_input, text="", onvalue=True, offvalue=False, variable=memory.reg_intercept, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=1, column=1, sticky="w")

        # --- x variables ---
        # create list of boolean tkinter variables to know if column is in model
        if memory.reg_x_bools == None:
            memory.reg_x_bools = [tk.BooleanVar(value=False) for _ in range(len(memory.df.columns))]
        
        tk.Label(self.frame_data_input, text="Vysvětlující proměnné (x):", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=2, column=0, sticky="w")
        
        # create table of checkboxes for all columns
        self.x_checkboxes = []
        self.x_lables = []
        if len(memory.df.columns) > 0:
            for i, col in enumerate(memory.df.columns):
                self.x_lables.append(
                    tk.Label(self.frame_data_input, text=col, font=memory.fonts["text"], bg=memory.bg_color_label)
                )
                self.x_checkboxes.append(
                    tk.Checkbutton(self.frame_data_input, text="", onvalue=True, offvalue=False, variable=memory.reg_x_bools[i], font=memory.fonts["text"], bg=memory.bg_color_checkbutton)
                )
                self.x_lables[i].grid(row=3+i, column=0, sticky="w")
                self.x_checkboxes[i].grid(row=3+i, column=1, sticky="w")
        
            self.disable_independent_variable_from_dependent()


        # --- assumption tests ---
        tk.Label(self.frame_right, text="Dodatečné nastaveníů", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        self.frame_assumptions_automatic = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_assumptions_automatic.pack(anchor="w")
        
        if memory.reg_manual_testing_bool == None:
            memory.reg_manual_testing_bool = tk.BooleanVar(value=False)

        tk.Label(self.frame_assumptions_automatic, text="Nastavit ručně:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(self.frame_assumptions_automatic, text="", onvalue=True, offvalue=False, variable=memory.reg_manual_testing_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton, command=self.manual_test_selection_update).grid(row=0, column=1, sticky="w")
        
        # grid frame for selection tests
        self.frame_assumptions = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_assumptions.pack(anchor="w")

        self.manual_test_selection_update()
        
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # selection of alpha value
        self.alpha_frame = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.alpha_frame.pack(anchor="w")
        tk.Label(self.alpha_frame, text="Hodnota alfa:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        self.textfield_alfa = tk.Text(self.alpha_frame, height=1, width=7, font=memory.fonts["text"], bg=memory.bg_color_label)
        self.textfield_alfa.insert(tk.INSERT, memory.regression_alpha)
        self.textfield_alfa.grid(row=0, column=1, sticky="w")



        tk.Button(self.frame_right, text="Provést lineární regresi", font=memory.fonts["text"], bg=memory.bg_color_button, command=self.calculate).pack(anchor="w")
        
        # runtime information - regression
        self.label_result_regression = tk.Label(self.frame_right, text="Lineární regrese: Nebyla provedena", bg=memory.bg_color_label)
        self.label_result_regression.pack(anchor="w")

        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"]).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)
    
    
    def manual_test_selection_update(self):
        
        tools.destroy_all_children(self.frame_assumptions)

        if self.root != None:
            tools.initial_resize_event(self.frame_assumptions)

        # hide test selection if manual testing is not selected
        if memory.reg_manual_testing_bool == None:
            return
        if memory.reg_manual_testing_bool.get() == False:
            return

        # test of homoskedasticity
        if memory.reg_test_hetero_bool == None:
            memory.reg_test_hetero_bool = tk.BooleanVar(value=True)
        tk.Label(self.frame_assumptions, text="Heteroskedasticita", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(self.frame_assumptions, text="Breusch-Paganův test", onvalue=True, offvalue=False, variable=memory.reg_test_hetero_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=0, column=1, sticky="w")

        # residual plot
        if memory.reg_residual_plot_bool == None:
            memory.reg_residual_plot_bool = tk.BooleanVar(value=True)
        tk.Label(self.frame_assumptions, text="Graf residuí", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=0, sticky="w")
        tk.Checkbutton(self.frame_assumptions, text="", onvalue=True, offvalue=False, variable=memory.reg_residual_plot_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=1, column=1, sticky="w")
        
        # normality test of residuals
        if memory.reg_test_normality_bool == None:
            memory.reg_test_normality_bool = tk.BooleanVar(value=True)
        tk.Label(self.frame_assumptions, text="Test normality", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=2, column=0, sticky="w")
        tk.Checkbutton(self.frame_assumptions, text="Shapirův-Wilkův test", onvalue=True, offvalue=False, variable=memory.reg_test_normality_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=2, column=1, sticky="w")
        

        # test of multicolinearity (vif)
        if memory.reg_test_vif_bool == None:
            memory.reg_test_vif_bool = tk.BooleanVar(value=True)
        tk.Label(self.frame_assumptions, text="Multikolinearita", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=3, column=0, sticky="w")
        tk.Checkbutton(self.frame_assumptions, text="Hodnoty VIF", onvalue=True, offvalue=False, variable=memory.reg_test_vif_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=3, column=1, sticky="w")

        # f-test
        if memory.reg_test_f_bool == None:
            memory.reg_test_f_bool = tk.BooleanVar(value=True)
        tk.Label(self.frame_assumptions, text="Celkový F-test", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=4, column=0, sticky="w")
        tk.Checkbutton(self.frame_assumptions, text="", onvalue=True, offvalue=False, variable=memory.reg_test_normality_bool, font=memory.fonts["text"], bg=memory.bg_color_checkbutton).grid(row=4, column=1, sticky="w")
        

        
        if self.root != None:
            tools.initial_resize_event(self.root)



    def calculate(self) -> None:
        
        # set recommended tests
        if memory.reg_manual_testing_bool.get() == False:
            print("Lineární regrese: související testy nastavené automaticky")
            memory.reg_test_hetero_bool = tk.BooleanVar(value=True) # turn on heteroskedasticity test
            memory.reg_test_vif_bool = tk.BooleanVar(value=True) # turn on vif calculation
            memory.reg_residual_plot_bool = tk.BooleanVar(value=True) # turn on plot generation
            memory.reg_test_normality_bool = tk.BooleanVar(value=True) # turn on s-w test of normality residuals
            memory.reg_test_f_bool = tk.BooleanVar(value=True) # turn on f-test
        else:
            print("Lineární regrese: související testy nastavené uživatelem")

        output = memory.get_empty_statistics_output_dict()

        alpha = tools.update_alfa_from_textbox(self.textfield_alfa)

        add_intercept = memory.reg_intercept.get()
        dependent_column = memory.reg_y.get()

        if not tools.check_variable(memory.df, dependent_column, require_number=True):
            output["runtime_result"] = "Regrese: nastala chyba (neplatná závislá proměnná)"
            output["valid_test_bool"] = False
            self.label_result_regression.configure(text=output["runtime_result"])
            return output

        independent_columns = []
        for i, col_bool in enumerate(memory.reg_x_bools):
            # if column shloud be included
            if col_bool.get():
                # get column name
                col_name = memory.df.columns[i]
                
                # if column is selected as dependent variable, ignore
                if col_name == dependent_column:
                    continue
                
                if not tools.check_variable(memory.df, col_name, require_number=True):
                    output["runtime_result"] = "Regrese: nastala chyba (neplatná nezávislá proměnná)"
                    output["valid_test_bool"] = False
                    self.label_result_regression.configure(text=output["runtime_result"])
                    return output

                # create list of dependent columns
                independent_columns.append(col_name)
        
        # number of parameters (independent variables) excluding intercept
        number_of_parameters_without_intercept = len(independent_columns)
        
        if number_of_parameters_without_intercept == 0:
            tools.error_message("Nebyla zvolena žádná nezávislá proměnná")
            output["runtime_result"] = "Regrese: nastala chyba"
            output["valid_test_bool"] = False
            return output

        # dataframe used to clean data
        df_regression = memory.df[independent_columns + [dependent_column]].copy()

        # handle NaN values
        df_regression.dropna(how="all", inplace=True)
        if df_regression.isnull().values.any():
            tools.error_message("Nelze provést lineární regresi, protože některý ze zvolených sloupců obsahuje neplatnou (NaN) hodnotu")
            return False

        if add_intercept:
            df_regression["Intercept"] = 1
            independent_columns.insert(0, "Intercept")

        # get matricies x and y from dataframe
        x = df_regression[independent_columns].values
        y = df_regression[dependent_column].values
        del df_regression

        # check for multicolinearity
        corr_mat = correlation.pearson(x, 0) # gets correlation matrix, but diagonal is replaced with 0
        if np.any(np.abs(corr_mat) > 0.999):
            output["multicolinearity_perfect"] = True
            tools.error_message(f"Regrese nemohla být provedena, protože některé nezávislé proměnné jsou dokonale korelované. Prosím odstraňtě některou proměnnou z modelu.")
            output["runtime_result"] = "Regrese: nastala chyba (dokonalá multikolinearita)"
            output["valid_test_bool"] = False
            return output
        else:
            output["multicolinearity_perfect"] = False
        
        # replace diagonal back with 1
        np.fill_diagonal(corr_mat, 1)

        # created dataframe from matrix (adds names of variables)
        corr_df = pd.DataFrame(corr_mat, columns=independent_columns)
        corr_df.insert(0, column="-", value=independent_columns)

        # calculate linear regression
        regression = RegressionLinear()
        error_singular, beta_coef, y_pred, residuals, r_squared = regression.fit_model(x, y, has_intercept=add_intercept, print_output=True) # calculates beta coefficients
        r_squared_adj = 1 - ( (1 - r_squared) * (len(y) - 1)) / (len(y) - number_of_parameters_without_intercept - 1)

        print(f"Upravené R^2: {r_squared_adj}")
        
        if error_singular:
            output["multicolinearity_perfect"] = True
            tools.error_message(f"Regrese nemohla být provedena, protože nezávislé proměnné tvoří lineární kombinaci. Prosím odstraňtě některou proměnnou z modelu.")
            output["runtime_result"] = "Regrese: nastala chyba (dokonalá multikolinearita)"
            output["valid_test_bool"] = False
            return output
        
        if add_intercept:
            output["equation"] = " + ".join(f"({round(coeff, memory.precision)}) * {var}" for coeff, var in zip(beta_coef[1:], independent_columns[1:]))
            output["equation"] = f"{round(beta_coef[0], memory.precision)} + " + output["equation"]
        else:
            output["equation"] = " + ".join(f"({round(coeff, memory.precision)}) * {var}" for coeff, var in zip(beta_coef, independent_columns))
        output["equation"] = f"{dependent_column} = " + output["equation"]
        
        print(f"Odhadnutá rovnice: {output['equation']}")


        stand_errors, t_statistics, p_values = regression.perform_t_tests(x, beta_coef, residuals)

        t_test_h0_denied_bools = p_values < alpha
        t_test_results_partial = np.where(t_test_h0_denied_bools, "H0 zamítnuta", "H0 nezamítnuta!")
        if np.all(t_test_h0_denied_bools):
            t_test_result_text = "Ve všech případech byla zamítnuta nulová hypotéza dilčích t-testů. Všechny koeficienty lze považovat za významné."
        else:
            t_test_result_text = "V alespoň jednom případě nebyla zamítnuta nulová hypotéza dilčího t-test, což značí, že některý z koeficientů je nevýznamný. Ověřte také přítomnost multikolinearity, neboť může značně ovlivnit výsledky dílčích t-testů."

        if memory.reg_test_f_bool.get():
            number_of_parameters_with_intercept = len(independent_columns) # number of parameters including intercept
            
            df_num = number_of_parameters_without_intercept
            df_den = len(y) - number_of_parameters_with_intercept
            
            sse = sum(residuals ** 2) # residual sum of squares
            
            if add_intercept:
                sst = sum((y - np.mean(y)) ** 2) # total sum of squares
            else:
                sst = sum(y ** 2) # total sum of squares
                
            ssr = sst - sse # regression sum of squares
            msr = ssr / df_num # regression mean square
            mse = sse / df_den # residual mean square
            f = msr / mse # observed f value

            

            f_critical = stats.f.ppf(1-alpha, df_num, df_den)
            p_value_f_test = 1 - stats.f.cdf(f, df_num, df_den)

            if p_value_f_test < alpha:
                reg_f_test_result = "Test zamítá nulovou hypotézu. Lze předpokládat, že alespoň jedna z vysvětlujících přoměných ovlivňuje vysvětlovanou."
            else:
                reg_f_test_result = "Test nezamítá nulovou hypotézu. Lze předpokládat, že ani jedna z vysvětlujících přoměných neovlivňuje vysvětlovanou."

            reg_f_test_table = pd.DataFrame({
                "Variabilita": ["Regresní", "Residuí", "Celková"],
                "Suma čtverců": [ssr, sse, sst],
                "Stupně volnosti": [df_num, df_den, df_num + df_den],
                "Průměrný čtverec": [msr, mse, ""],
                "F-statistika": [f, "", ""],
                "F-kritická": [f_critical, "", ""],
                "P-hodnota": [p_value_f_test, "", ""]
            })
            
            print("\nF-test:")
            print(reg_f_test_table)

            output["f_test_table"] = reg_f_test_table
            output["f_test_result"] = reg_f_test_result




        # Breusch-Pagan homoskedasticity test
        if memory.reg_test_hetero_bool.get():
            hetero_statistic, hetero_critical, hetero_p_value, hetero_result_text = regression.perform_breusch_pagan_test(x, residuals, alpha, number_of_parameters_without_intercept, add_intercept)
            output["hetero_statistic"] = hetero_statistic
            output["hetero_critical"] = hetero_critical
            output["hetero_p_value"] = hetero_p_value
            output["hetero_result_text"] = hetero_result_text
            output["hetero_valid_output"] = True
        else:
            output["hetero_valid_output"] = False

        # varience influece facotrs (VIF)
        if memory.reg_test_vif_bool.get():
            vif_high = False
            vifs = []
            vif_lables = []
            
            for i in range(len(independent_columns)):
                
                y_vif = x[:, i]
                x_vif = np.delete(x, i, axis=1)
                
                error_singular, _, _, _, r_squared_vif = regression.fit_model(x_vif, y_vif, has_intercept=add_intercept)
                
                vif = 1 / (1 - r_squared_vif)

                vifs.append(vif)
                
                if vif > 5:
                    vif_lables.append("Ano")
                    vif_high = True

                else:
                    vif_lables.append("Ne")
        
            vif_table = pd.DataFrame(data={"Proměnná": independent_columns, "VIF": vifs, "Multikolinearita (VIF > 5)": vif_lables})

            print("VIF:")
            print(vif_table)
        

        # normality of residuals s-w test
        if memory.reg_test_normality_bool.get():
            normality_df = pd.DataFrame({"data_internal_column": residuals})
            normality_test = normality.Normality(normality_df, selected_test="Shapirův-Wilkův test")
            normality_output = normality_test.calculate(alpha)

            normality_statistic = normality_output["statistic"][0]
            normality_p_value = normality_output["p_value"][0]
            normality_accurate = normality_output["accurate"][0] # if test meets requirement of < 5000 observations
            normality_requirement = normality_output["requirement"]
            
            if normality_p_value < alpha:
                normality_result = "Byla zamítnuta nulová hypotéza. Rozdělení residuí nepochází z normálního rozdělení, čímž je porušen předpoklad lineární regrese."
            else:
                normality_result = "Nebyla zamítnuta nulová hypotéza. Rozdělení residuí by mohlo pocházet z normálního rozdělení, čímž není porušen předpoklad lineární regrese."

            if not normality_accurate:
                normality_result = normality_result + f" Tento závěr však nemusí být přesný, neboť byl porušen předpoklad ({normality_requirement}) testu normality."


            output["norm_statistic"] = normality_statistic
            output["norm_p_value"] = normality_p_value
            output["norm_result"] = normality_result
            output["norm_valid_output"] = True
        else:
            output["norm_valid_output"] = False


        result_table = pd.DataFrame({
            "Proměnná": independent_columns,
            "Koeficient": beta_coef,
            "Standard error": stand_errors,
            "T-statistika": t_statistics,
            "P-hodnota": p_values,
            "T-test": t_test_results_partial
        })

        print("\nVýsledky lineární regrese:")
        print(result_table)

        
        output["valid_test_bool"] = True    
        output["valid_output"] = True
        output["runtime_result"] = "Regrese: provedena úspěšně"
        output["result_table"] = result_table
        output["r_squared"] = r_squared
        output["r_squared_adj"] = r_squared_adj
        output["correlation_df"] = corr_df
        output["residuals_standardized"] = tools.standardize_vector(residuals)
        output["y"] = y
        output["y_name"] = dependent_column
        output["t_test_result"] = t_test_result_text
        if memory.reg_test_vif_bool.get():
            output["vif_table"] = vif_table
            if vif_high:
                output["vif_result"] = "Některé proměnné mají vysokou hodnotu VIF, což indikuje přítomnost multikolinearity. Pro správnou interpretaci regresních koeficientů zvažte vynechání některé proměnné z modelu."
            else:
                output["vif_result"] = "Všechny proměnné mají nízkou hodnotu VIF, což neindikuje multikolinearitu."

        memory.reg_output = output

        self.label_result_regression.configure(text=output["runtime_result"])

        # calculate homogenity
        #self.homogenity = homogenity.Homogenity(self.df_reg, self.homogenity_test_var.get())
        #homogenity_output = self.homogenity.calculate(alfa_float)
        #self.label_result_homogenity.config(text=homogenity_output["runtime_result"])
        #if homogenity_output["valid_test_bool"] != False: memory.anova_homogenity = homogenity_output # homogenity sucessful
 


class RegressionOutput():
    
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
        
        # header
        tk.Label(self.frame_right, text="Lineární regrese - výstup", font=memory.fonts["header_big"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # -- independece --
        self.frame_requirements = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_requirements.pack(anchor="w")
        tk.Label(self.frame_requirements, text="Předpoklad nezávislosti pozorování", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        label_independence = tk.Label(self.frame_requirements, text="Model lineární regrese předpokládá, že jsou jednotlivá pozorování ve vstupních datech na sobě nezávislá. Tento předpoklad lze nejlépe ověřit na základě znalosti metody sběru dat.", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        label_independence.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_independence)

        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        # -- normal distribution of residuals --
        tk.Label(self.frame_requirements, text="Předpoklad normálního rozdělení náhodné složky", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        label_normality = tk.Label(self.frame_requirements, text="Regresní model předpokládá, že náhodná složka odpovídá normálnímu rozdělení.", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        label_normality.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_normality)

        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")


        # -- perfect multicolinearity --
        tk.Label(self.frame_requirements, text="Předpoklad nedokonalé multikolinearity", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        label_multicolinearity = tk.Label(self.frame_requirements, text="Regresní model nesmí obsahovat dokonalou multikolinearitu, jinak není možné odhadnout regresní parametry klasickou metodou nejmenších čtverců", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        label_multicolinearity.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_multicolinearity)

        if "correlation_df" in memory.reg_output:
            tk.Label(self.frame_requirements, text="Dokonalá multikolinearita nebyla detekovaná", font=memory.fonts["header"], bg=memory.bg_color_label, justify="left", fg=memory.fg_color_label_highlight).pack(anchor="w")
            tk.Label(self.frame_requirements, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_requirements, text="Korelační matice", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
            label_corelation_explained = tk.Label(self.frame_requirements, text="Níže je uvedená korelační matice, která udává hodnoty Pearsonova korelačního koeficientu pro každou z dvojic vysvětlujících proměnných. V případě, že jsou některé proměnné navzájem silně korelované (hodnota blízká 1 nebo -1), jedná se o indikaci přítomnosti multikolinearity.", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_corelation_explained.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_corelation_explained)
            frame_correlation_table = tk.Frame(self.frame_requirements, bg=memory.bg_color_frame)
            frame_correlation_table.pack(anchor="w")
            tools.LabelsTable(frame_correlation_table, memory.reg_output["correlation_df"], memory.fonts["text"], memory.fonts["text"], precis, True)

        # this will never run, because in case of perfect multicolinearity, it throws error
        #if "multicolinearity_perfect" in memory.reg_output:
        #    if memory.reg_output["multicolinearity_perfect"]:
        #        label_multicolinearity_perfect_result = tk.Label(self.frame_requirements, text="Některé nezávislé proměnné jsou dokonale lineárně závislé a proto nelze odhadnou správně regresní parametry", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        #        label_multicolinearity_perfect_result.pack(anchor="w")
        #        self.frame_right.wrappable_labels.append(label_multicolinearity_perfect_result)

        

        if "vif_table" in memory.reg_output:
            tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_right, text="VIF (variance inflation factors)", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
            label_vif_explained = tk.Label(self.frame_right, text="Multikolinearita způsobuje nestabilní odhady regresních parametrů, což má za následek zkreslení dílčích t-testů a problémy s interpretací modelu. Variační inflační koeficienty VIF udávají míru, nakolik daná proměnná přispívá k rozptylu regresních koeficientů. Hodnoty větší než 5 značí vliv multikolinearity, přičemž hodnoty větší než 9 značí silný vliv.", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_vif_explained.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_vif_explained)
            frame_vif_table = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
            frame_vif_table.pack(anchor="w")
            tools.LabelsTable(frame_vif_table, memory.reg_output["vif_table"], memory.fonts["header"], memory.fonts["text"], round_values=precis, first_col_allign_left=True)
            label_vif_result = tk.Label(self.frame_right, text=memory.reg_output["vif_result"], font=memory.fonts["header"], bg=memory.bg_color_label, justify="left", fg=memory.fg_color_label_highlight)
            label_vif_result.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_vif_result)

        # --- f test table ---
        if "f_test_table" in memory.reg_output:
            tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_right, text="Celkový F-test", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
            self.frame_reg_f_test_table = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
            self.frame_reg_f_test_table.pack(anchor="w")
            
            if isinstance(memory.reg_output["f_test_table"], pd.DataFrame):
                tools.LabelsTable(self.frame_reg_f_test_table, memory.reg_output["f_test_table"], memory.fonts["header"], memory.fonts["text"], round_values=precis, first_col_allign_left=True)

            label_f_test_result = tk.Label(self.frame_right, text=memory.reg_output["f_test_result"], font=memory.fonts["header"], bg=memory.bg_color_label, justify="left", fg=memory.fg_color_label_highlight)
            label_f_test_result.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_f_test_result)

        # --- regression model table ---
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Regresní model", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        if "r_squared" in memory.reg_output:
            tk.Label(self.frame_right, text=f"Koeficient determinace (R^2): {round(memory.reg_output['r_squared'], precis)}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_right, text=f"Upravený koeficient determinace (R^2 adjusted): {round(memory.reg_output['r_squared_adj'], precis)}", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            tk.Label(self.frame_right, text=f"Model dokáže vysvětlit přibližně {round(100 * memory.reg_output['r_squared'])}% variability", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
            label_t_test_result = tk.Label(self.frame_right, text=memory.reg_output["t_test_result"], font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
            label_t_test_result.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_t_test_result)
            label_reg_equation = tk.Label(self.frame_right, text="Odhadnutý vztah:", font=memory.fonts["text"], bg=memory.bg_color_label)
            label_reg_equation.pack(anchor="w")
            self.frame_right.wrappable_labels.append(label_reg_equation)
            tk.Label(self.frame_right, text=memory.reg_output["equation"], font=memory.fonts["header"], bg=memory.bg_color_label, fg=memory.fg_color_label_highlight).pack(anchor="w")

        # --- regression model table ---
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        self.frame_reg_table = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
        self.frame_reg_table.pack(anchor="w")

        if "result_table" in memory.reg_output:
            if isinstance(memory.reg_output["result_table"], pd.DataFrame):
                tools.LabelsTable(self.frame_reg_table, memory.reg_output["result_table"], memory.fonts["header"], memory.fonts["text"], round_values=precis, first_col_allign_left=True)

        
        
        # --- heteroskedasticity test ---
        if "hetero_valid_output" in memory.reg_output:
            if memory.reg_output["hetero_valid_output"]:
                tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                tk.Label(self.frame_right, text="Test heteroskedasticity", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
                tk.Label(self.frame_right, text="Pro detekci heteroskedasticity byl použit Breusch-Paganův test", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                tk.Label(self.frame_right, text="H0: Rozptyl residuí je konstantní", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                tk.Label(self.frame_right, text="H1: Rozptyl residuí není konstantní", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                
                frame_hetero = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
                frame_hetero.pack(anchor="w")

                tk.Label(frame_hetero, text="Statistika:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
                tk.Label(frame_hetero, text="Kritická hodnota:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=0, sticky="w")
                tk.Label(frame_hetero, text="P-hodnota:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=2, column=0, sticky="w")

                tk.Label(frame_hetero, text=str(round(memory.reg_output['hetero_statistic'], precis)), font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=1, sticky="w")
                tk.Label(frame_hetero, text=str(round(memory.reg_output['hetero_critical'], precis)), font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=1, sticky="w")
                tk.Label(frame_hetero, text=str(round(memory.reg_output['hetero_p_value'], precis)), font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=2, column=1, sticky="w")

                label_hetero_result_text = tk.Label(self.frame_right, text=memory.reg_output['hetero_result_text'], justify="left", bg=memory.bg_color_label, fg=memory.fg_color_label_highlight, font=memory.fonts["header"])
                label_hetero_result_text.pack(anchor="w")
                self.frame_right.wrappable_labels.append(label_hetero_result_text)

        # residuals normality test
        if "norm_valid_output" in memory.reg_output:
            if memory.reg_output["norm_valid_output"]:
                tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                tk.Label(self.frame_right, text="Test normality residuí", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
                tk.Label(self.frame_right, text="Pro test normality residuí byl použit Shapirův-Wikův test", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                tk.Label(self.frame_right, text="H0: Náhodná pozorování pocházejí z normálního rozdělení", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                tk.Label(self.frame_right, text="H1: Náhodná pozorování nepocházejí z normálního rozdělení", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                
                frame_norm = tk.Frame(self.frame_right, bg=memory.bg_color_frame)
                frame_norm.pack(anchor="w")

                tk.Label(frame_norm, text="Statistika:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=0, sticky="w")
                tk.Label(frame_norm, text="P-hodnota:", font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=0, sticky="w")

                tk.Label(frame_norm, text=str(round(memory.reg_output['norm_statistic'], precis)), font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=0, column=1, sticky="w")
                tk.Label(frame_norm, text=str(round(memory.reg_output['norm_p_value'], precis)), font=memory.fonts["text"], bg=memory.bg_color_label).grid(row=1, column=1, sticky="w")

                label_norm_result_text = tk.Label(self.frame_right, text=memory.reg_output['norm_result'], bg=memory.bg_color_label, font=memory.fonts["header"], fg=memory.fg_color_label_highlight, justify="left")
                label_norm_result_text.pack(anchor="w")
                self.frame_right.wrappable_labels.append(label_norm_result_text)
        
        # plot residuals
        if memory.reg_residual_plot_bool != None:
            if memory.reg_residual_plot_bool.get():
                tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
                tk.Label(self.frame_right, text="Graf residuí", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
                label_residual_plot_explained = tk.Label(self.frame_right, text="Zde je zobrazen graf residuí. Jendá se o nevysvětlenou odchylku pozorovaných hodnot od predikovaných. V případě, že je v grafu zřetelný vztah mezi velikostí odchylky a vysvětlovanou proměnnou, nejspíše nebyl zvolen vhodný model.", bg=memory.bg_color_label, justify="left", font=memory.fonts["text"])
                label_residual_plot_explained.pack(anchor="w")
                self.frame_right.wrappable_labels.append(label_residual_plot_explained)
                fig_residuals = Figure(figsize=memory.fig_size, dpi=memory.fig_dpi)
                plot_residuals = fig_residuals.add_subplot()
                plot_residuals.scatter(memory.reg_output["y"], memory.reg_output["residuals_standardized"])
                plot_residuals.axhline(0, color='red', linestyle='--')
                plot_residuals.set_ylabel("Standardizovaná residua")
                plot_residuals.set_xlabel(memory.reg_output["y_name"])
                plot_residuals.set_title("Graf residuí")
                y_limit = max(np.abs(memory.reg_output["residuals_standardized"])) + 0.5
                plot_residuals.set_ylim(-y_limit, y_limit)
                tools.plot(fig_residuals, self.frame_right, include_toolbar=True)


        # empty label to prevent clipping of edge with last row of text
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)
        #tools.initial_resize_event(self.frame_right)