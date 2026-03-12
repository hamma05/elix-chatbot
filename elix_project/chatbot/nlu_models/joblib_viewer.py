#!/usr/bin/env python3
import numpy as np
import joblib
import sys
import textwrap
import py_cui
from functools import partial

def get_str_type(value):
    if isinstance(value, np.ndarray):
        return(f'{type(value)} of shape {value.shape}')
    else:
        return type(value)

menu_width = 4
    
jdata = joblib.load('intent_en_v1.joblib')

root = py_cui.PyCUI(2, 10)

class MenuOperator():
    def __init__(self, data):
        self.data = data
        self.stack = []
        self.select = []
        self.menu = root.add_scroll_menu("Title", 0, 0, row_span = 1, column_span=menu_width, padx = 1, pady = 0)
        self.text = root.add_text_block("Info", 0, menu_width, row_span=2, column_span=10-menu_width)
        self.text.set_selectable(False)
        self.update()
        self.menu.add_key_command(py_cui.keys.KEY_ENTER, self.my_function)
        self.menu.add_key_command(py_cui.keys.KEY_DOWN_ARROW, partial(self.preview, par=1))
        self.menu.add_key_command(py_cui.keys.KEY_UP_ARROW, partial(self.preview, par=-1))

    def update(self):
        self.select = []
        for key, value in self.data.items():
            self.select.append('%s : %s' % (key, get_str_type(value)))
        self.select.append('Quit')
        self.menu.clear()
        self.menu.add_item_list(self.select)

    def my_function(self):
        in_prog = self.menu.get()
        if in_prog == 'Quit':
            if len(self.stack) == 0:
                sys.exit()
            else:
                self.data = self.stack.pop()
                self.update()
                return
        self.text.clear()
        key = in_prog.split(':')[0].strip()
        if isinstance(self.data[key], dict):
            self.stack.append(self.data)
            self.data = self.data[key]
            self.update()
        else:
            self.text.write('\n'.join(textwrap.wrap(str(self.data[key]), width=self.text._width)))

    def preview(self, par):
        in_prog = self.menu.get()
        #print(in_prog, select)
        #import ipdb; ipdb.set_trace()
        pos = self.select.index(in_prog) + par
        if pos >= len(self.select):
            return
        in_prog = self.select[pos]
        self.text.clear()
        key = in_prog.split(':')[0].strip()
        if key == 'Quit':
            return
        self.text.write(self.format(self.data[key]))

    def format(self, obj):
        if isinstance(obj, dict):
            lines = []
            for key, value in obj.items():
                lines.append('%s : %s' % (key, value))
            lines = '\n'.join(lines)
            #import ipdb; ipdb.set_trace()
            return lines #'\n'.join(textwrap.wrap(lines, width=self.text._width))
        else:
            return '\n'.join(textwrap.wrap(str(obj), width=self.text._width))
mo = MenuOperator(jdata)
root.set_selected_widget(mo.menu)
root._in_focused_mode=True

root.start()
