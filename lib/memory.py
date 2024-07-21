

print("Načítání modulu: Paměť nastavení")

import pandas as pd

def get_empty_statistics_output_dict() -> dict:
    
    output = {
        "p_value": 0,
        "statistic": 0,
        "date_time": "test zatím nebyl proveden",
        "alpha": "0.05",
        "test_used": "---",
        "h0": "---",
        "h1": "---",
        "result": "---",
        "runtime_result": "---",
        "result_anova": "---",
        "valid_test_bool": True, # if real test was selected
        "result_table": None,
        "valid_output": False
        }
    
    return output





fonts = {}

dataset_selected = False
file_source = ""

fig_size = (6, 4)
fig_dpi = 100

precision = 3

import_extention = None
import_includes_header = None
import_delimieter = None

df = pd.DataFrame()

bg_color_default = "alice blue"

bg_color_label = bg_color_default
bg_color_frame = bg_color_default
bg_color_checkbutton = bg_color_default
bg_color_radiobutton = bg_color_default
bg_color_button = "light sky blue"
bg_color_text = bg_color_default
bg_color_scroll_frame = bg_color_default
bg_color_label_in_table = bg_color_default
bg_color_label_in_table_header = "skyblue3"
bg_color_scroll_container = bg_color_default
bg_color_scroll_canvas = bg_color_default
bg_color_label_left_panel = bg_color_default
bg_color_button_left_panel = "seashell2"
bg_color_panned_window_frame_border = "seashell2"
fg_color_label_highlight = "skyblue4"

datatype_options = ["Celá čísla", "Desetinná čísla", "Kategorie (text)", "Logická hodnota (ano/ne)", "Datum a čas"]

homogenity_tests = ["Bartlettův test", "Levenův test"]
normality_tests = ["Shapirův-Wilkův test"]
mct_tests = ["Tukeyho test", "Scheffeho test"]

banned_column_name_list = ["_count_internal_column", "Očekávané četnosti", "Pozorované četnosti", "Suma", "Intercept", "---", "data_internal_column"]
banned_columns_string = "\n\nSeznam zakázaných názvů sloupců:\n" + "\n".join(banned_column_name_list)

# test outputs
anova_homogenity = get_empty_statistics_output_dict()
anova_normality = get_empty_statistics_output_dict()
anova_mct = get_empty_statistics_output_dict()
anova_test = get_empty_statistics_output_dict()
anova_manual_testing = None
anova_qq_plots_list = None
anova_boxplot_figure = None

reg_output = get_empty_statistics_output_dict()
reg_manual_testing_bool = None
reg_test_hetero_bool = None
reg_test_vif_bool = None
reg_residual_plot_bool = None
reg_test_normality_bool = None
reg_test_f_bool = None

# test settings
anova_test_normality_bool = None
anova_test_homogenity_bool = None
anova_test_mct_bool = None
anova_test_homogenity_var = None
anova_test_mct_var = None
anova_qq_plot_bool = None
anova_boxplot_bool = None

# selected variables - contingency
cont_data_input_col_name_factor_1 = None
cont_data_input_col_name_factor_2 = None
cont_data_input_col_name_weight = None

# test settings - correlation
corr_pearson_bool = None
corr_spearman_bool = None
corr_scatter_plot_bool = None
corr_variable_bools = None # list of selected variables

# correlation output
corr_output = get_empty_statistics_output_dict()

contingency_output = get_empty_statistics_output_dict()

data_input_col_names = None


test_list = []




regression_alpha = 0.05

def reset_memory():
    """Sets variables to none in order to 'reset' them. Used to clear previously saved setting usually when input dataframe is changed.
    """
    print("Načítání původního nastavení...")
    
    global reg_x_bools, reg_intercept, reg_y, contingency_include_weight_col, corr_variable_bools, anova_test_normality_bool
    global anova_test_homogenity_bool, anova_test_mct_bool, anova_test_homogenity_var, anova_test_homogenity_var, anova_test_mct_var, anova_qq_plot_bool, anova_boxplot_bool, corr_pearson_bool
    global corr_spearman_bool, corr_scatter_plot_bool, corr_output, contingency_output, anova_homogenity, anova_normality, anova_mct, anova_test, anova_manual_testing, anova_qq_plots_list, anova_boxplot_figure
    global reg_output, anova_data_input_col_names
    global cont_data_input_col_name_factor_1, cont_data_input_col_name_factor_2, cont_data_input_col_name_weight

    reg_x_bools = None # list of tkinter bools wether column is in model or not
    reg_intercept = None # tkinter bool wether intercept shloud be added to model
    reg_y = None # tkinter string selected dependent variable

    corr_variable_bools = None

    corr_output = get_empty_statistics_output_dict()
    contingency_output = get_empty_statistics_output_dict()

    anova_homogenity = get_empty_statistics_output_dict()
    anova_normality = get_empty_statistics_output_dict()
    anova_mct = get_empty_statistics_output_dict()
    anova_test = get_empty_statistics_output_dict()
    anova_qq_plots_list = None
    anova_boxplot_figure = None
    anova_data_input_col_names = None

    cont_data_input_col_name_factor_1 = None
    cont_data_input_col_name_factor_2 = None
    cont_data_input_col_name_weight = None

    reg_output = get_empty_statistics_output_dict()


    contingency_include_weight_col = None


