
print("..........................")
print("... Spouštění programu ...")
print("..........................")

print()
print("Čekejte prosím ...")


from lib import tools
from lib import data_input
from lib import memory
from lib import anova
from lib import contingency
from lib import regression
from lib import introduction
from lib import correlation


import tkinter as tk
import base64
from tkinter import font

import sys

import xlrd
import openpyxl

if getattr(sys, "frozen", False):
    import pyi_splash # splash screen (only for compiled version to exe)

print("Načítání modulů dokončeno")

to_resize = [] # list off all objects that have functiion ".resize"

# Symanalytics = symbiotic analysis = symbiotic, because it works in symbiosis with the user - without the user the software is useless as it can not interpret it's outputs

root = tk.Tk("Symanalytics")
root.title("Symanalytics")

root.resizable(True, True) 
root.geometry("1100x900")
#root.state("zoomed") # maximalize window

# initialize some memory variables
memory.reset_memory()

memory.fonts = {
    "header": font.Font(family="Arial black", size=10, weight="bold"),
    "header_left_panel": font.Font(family="Arial black", size=12, weight="bold"),
    "header_huge": font.Font(family="Arial black", size=12, weight="bold"),
    "header_big": font.Font(family="Arial black", size=11, weight="bold"),
    "text": font.Font(family="Arial", size=10, weight="normal", overstrike=False),
    "text_disabled": font.Font(family="Arial", size=10, weight="normal", overstrike=True)
}


# for .exe file, image (png) must me converted to 64bit string and than decoded as base64 data
icon_file_as_string = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAG8UExURf///4ODg2NjY2JiYoCAgO3t7YmJiQAAAFVVVTQ0NOjo6NLS0pSUlH5+fkBAQCoqKt3d3cjIyB0dHXR0dKSkpDU1NSAgIDo6Or6+vri4uCwsLH9/f2pqap+fnysrKxYWFkhISLS0tLu7uzIyMmBgYJubm/39/QwMDFZWVr29vePj46qqqjg4OGlpafv7+4GBgRkZGaCgoBUVFff394uLiwgICPPz8wICAllZWbKyssvLy5OTk1dXVwcHBxoaGs/Pz2VlZX19fW5ubmtrawoKCkdHR9vb27+/v6Ojo4+Pj3x8fFBQUDs7OxgYGIeHh7e3t1RUVHh4eOfn5wEBAampqe/v76+vr7y8vLq6uqioqJ2dnQQEBDAwMHBwcKenp9/f34yMjJGRkTc3N0lJSdnZ2TMzM0JCQuTk5JCQkObm5tjY2AkJCTY2NtXV1erq6pKSktPT04KCgkZGRj4+Pj09Pdra2vLy8mxsbFNTU2hoaOLi4tfX197e3uvr69TU1MfHxxQUFDw8PI6OjgUFBQMDAxAQECQkJMPDw1hYWJeXl1xcXBwcHExMTERERGRkZLOzswsLCygoKAYGBqurq5vstBEAAAAJcEhZcwAADsIAAA7CARUoSoAAAAM9SURBVFhH1ZbrQxJBFMW3l5NplFlYkEUGaWZkomia0sNEJUAw6AGSPext2dOMTMlQpNSsf7idO4cVkMcO3/p92L337Jy7C3tnZpX/hh07d6ns3oNUmiom2ItcmmoUYPsgSFIDO2O1UCTZDztjBiiSHIBd5SAkKerIeoiOVdCkqCfrYToegSaFkTsbFCrAaiBKcJSMxxQTnc1QJThORkVppPMJqBKc5D6LGlABVidU/ZwiW5ManaaoXsj6sZKNRzaKjKTq50wzd7VQTAXYWYqJVpxLcS7L1EbxeYoVxX6hXc/vaSGTiC9S3EGxo7NLjfl/WxpnN/f0IOMxY5ccvX2X+3k04MKF4oibXrnKuXa9nbJBOqrccGBUPo4h9/DIqNXqMdykGxWmg7/abTjcZosXI0oy6Ct0e/8YvbTyeANBWLLxi+mSzTjOedyyw5JNyIOrwOvxhW9T1HZH4y4J7B482QQHxDVOJDoRu8/FSUof0ADBQ1IeIcvisfjxXZapJwG/EyJa7ykyzjNS2HOkGi9ecvmVTbMS0zS4ARnxmiT2BqkGrVpsBlmGt6ROIhO8I+09sgxBUlkYaYYPpH5EJhBzmn1CCsSzslmkGWhH+4wEOOlh+/PbYIqrrHsuhBx8iX/NU1Tmv8W3b5ELVICxRRsEaUZQgSXcUGT5vqS6f/AShgpLuJLcTc1viBWb6CVp7enQSngD5deaAjjN9BSC5cAKZBlCnaInBcnUahwX9GNvSvPlVuNnBUWcw78isAu86VjuLCvPvG0t5zlYZC1vppbHGV7PXV2brfIN0mvWvhCJaP581UGrzUL9CaIb0GVwTDemFlFAbdOKWkyxuye0DrHq2dUL4RpFjd++QvuCLtxWPmkZ21yAIEs4Sn515dm+UpXHOUx2tUv/sCQ0/QRjadp7lviErRUbl16G/H1zm9ys3pq+F9ZxoRyh3vDIhMe0NSnEXp3sw/XiOFx/zanqrd7Jxugr/ffNxAIePG4BTKVWKbvLNpvI+UAZF+8cLKcaS3VgKJU1/8fzPkw2x3wb8xhYjBjG5hIxjQViLl1rfF4BbyI9anPLvOwVYTQm0oFVf7ySfWWjM7ZS6TStHEX5B7LgjK4FQd68AAAAAElFTkSuQmCC"
icon_base_64_decoded = base64.b64decode(icon_file_as_string)
icon_image = tk.PhotoImage(data = icon_file_as_string)
root.wm_iconphoto(True, icon_image)
del icon_file_as_string, icon_base_64_decoded

root.update() # update in order to correctly get window height and width

# creates panned window that will be filled with left and right frames
main_panned_window = tk.PanedWindow(root, bd=1, bg=memory.bg_color_panned_window_frame_border)
main_panned_window.pack(fill="both", expand=1)

frame_left = tools.create_scrollable_frame(main_panned_window, stretch_width=True)
frame_right = tools.create_scrollable_frame(main_panned_window, stretch_width=False)

main_panned_window.sash_place(0, 250, 1) # place the devider (sash) of panned window at x position 250

# left panel - data input
tk.Label(frame_left, text="Úvod:", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")
intro_module = introduction.Introduction(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (data input)

# empty label
tk.Label(frame_left, text="", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")

# left panel - data input
tk.Label(frame_left, text="Vstupní data:", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")
data_input.DataInputSelector(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (data input)
data_input.DataInputFormater(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (data format)

# empty label
tk.Label(frame_left, text="", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")

# left panel - anova
tk.Label(frame_left, text="Analýza rozptylu:", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")
anova.AnovaInput(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (anova input)
anova.AnovaOutput(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (anova output)

# empty label
tk.Label(frame_left, text="", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")

# left panel - contingency table
tk.Label(frame_left, text="Kontingenční tabulka:", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")
contingency.ContingencyInput(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (contingency input)
contingency.ContingencyOutput(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (contingency output)

# empty label
tk.Label(frame_left, text="", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")

# left panel - correlation
tk.Label(frame_left, text="Korelační analýza:", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")
correlation.CorrelationInput(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (correlation input)
correlation.CorrelationOutput(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (correlation output)

# empty label
tk.Label(frame_left, text="", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")


# left panel - regression
tk.Label(frame_left, text="Regrese:", font=memory.fonts["header_left_panel"], bg=memory.bg_color_label_left_panel).pack(anchor="w", fill="x")
regression.RegressionInput(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (regression input)
regression.RegressionOutput(frame_left, frame_right, root) # creates button in left panel with all functionality integrated (regression output)


# load introduction window first
intro_module.load_panel()


# forces configure update when the whole windows changes
# (fixes bug - forces rescaling of main panned window when user uses Windows button to maximize window)
def update_all_windows_size(event) -> None:
    if event.widget.winfo_id() == root.winfo_id():
        main_panned_window.update()
        #print("main_panned_window updated")
        
        #print(main_panned_window.winfo_children())
        for child_widget in root.winfo_children():
            child_widget.event_generate("<Configure>")
root.bind("<Configure>", update_all_windows_size)
#main_panned_window.bind("<Configure>", update_all_windows_size)


# close splash screen (only for compiled version to exe)
if getattr(sys, "frozen", False):
    pyi_splash.close()

root.mainloop()