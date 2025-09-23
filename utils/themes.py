import dearpygui.dearpygui  as dpg


# создаём зелёную тему
with dpg.theme() as green_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(
            dpg.mvThemeCol_Button, (0, 200, 0, 255), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonHovered, (0, 220, 0, 255), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonActive, (0, 160, 0, 255), category=dpg.mvThemeCat_Core
        )

# создаём красную тему
with dpg.theme() as red_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(
            dpg.mvThemeCol_Button, (200, 0, 0, 255), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonHovered, (220, 0, 0, 255), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonActive, (160, 0, 0, 255), category=dpg.mvThemeCat_Core
        )
