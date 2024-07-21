
print("Načítání modulu: Úvod")

from lib import tools
from lib import memory

import tkinter as tk


class Introduction():
    def __init__(self, frame_left, frame_right, root = None) -> None:        
        """
        Creates button in frame_left that after clicking fill frame_right with data input options.

        frame_left: tkinter widget
            Parent left frame
        frame_right: tkinter widget
            Parent right frame
        root: tkinter widget
            Parent widget on which resize event will be called to properly resize widgets (fixes bug with resizing scrollbars)
        """

        self.frame_right = frame_right
        self.root = root

        # left panel button - data format
        tk.Button(frame_left, text="Úvodní informace", command=self.load_panel, font=memory.fonts["text"], bg=memory.bg_color_button_left_panel).pack(anchor="nw", fill="x")

    def load_panel(self) -> None:
        
        tools.destroy_all_children(self.frame_right)

        # choose file path
        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")
        tk.Label(self.frame_right, text="Úvod:", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")
        
        label_intro = tk.Label(self.frame_right, text="Výtejte v programu Symanalytics. Tento program vznikl v rámci bakalářské práce na Fakultě informatiky a statistiky VŠE. Jedná se o program, který dokáže provést základní statistické testy a analýzy.", font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        label_intro.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_intro)

        tk.Label(self.frame_right, text="Vytvořil Šimon Charvát v roce 2024", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        tk.Label(self.frame_right, text="", font=memory.fonts["text"], bg=memory.bg_color_label).pack(anchor="w")

        tk.Label(self.frame_right, text="Použití programu", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")

        label_usage = tk.Label(self.frame_right, text='Program je vertikálně rozdělen na dvě části. Levý postranní panel slouží k přepínání oken, zatímco zde na pravé straně se objevují příslušná okna. Pro provedení analýzy je nejprve nutné nahrát data, což provedete pomocí tlačítka "Nahrát soubor". Data je možné nahrát pomocí souboru typu Excel či CSV. Po jeho nahrání je možné přejít k provádění analýz samotných, které je nutné vybrat opět pomocí levého postranního panelu.', font=memory.fonts["text"], bg=memory.bg_color_label, justify="left")
        label_usage.pack(anchor="w")
        self.frame_right.wrappable_labels.append(label_usage)

        tk.Label(self.frame_right, text="", font=memory.fonts["header"], bg=memory.bg_color_label).pack(anchor="w")

        if self.root != None:
            tools.initial_resize_event(self.root)


