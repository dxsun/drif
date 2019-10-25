import PySimpleGUI as sg
import time

nl_command_file = "tmp_instr.txt"
nl_real_file = "tmp_instr_real.txt"


def write_command_file(nl_command):
    with open(nl_command_file, "w") as fp:
        fp.write(nl_command)
    print("Command <" + str(nl_command) + "> sent!")

def read_suggest_file():
    with open(nl_real_file, "r") as fp:
        real_nl = fp.read()
    return real_nl or ""

def b1(): 
    return sg.Text("    Enter:", font=docfont)

def b1b():
    return sg.Text("Execute command", font=docfont)

def b2():
    return sg.Text("    Left:  ", font=docfont)

def b2b():
    return sg.Text("Reset the drone to starting position", font=docfont)

def b3():
    return sg.Text("    Right:", font=docfont)

def b3b():
    return sg.Text("Go to the next environment", font=docfont)

def b4():
    return sg.Button('OK', font=buttonfont, button_color=buttoncolor)
        
def b5():
    return sg.Button('Next',  font=buttonfont, button_color=buttoncolor)

def b6():
    return sg.Button("Reset",  font=buttonfont, button_color=buttoncolor)

def b7():
    return sg.Button('Clear Text',  font=buttonfont, button_color=buttoncolor)

if __name__ == "__main__":
    keep_going = True
    def b_form():
        return sg.FlexForm("Enter the navigation command",
            return_keyboard_events=True,
            default_element_size=(90, 40))
    form = b_form()
    inputfont = ('Helvetica 30')
    buttonfont = ('Helvetica 20')
    buttoncolor = ("#FFFFFF", "#333333")
    docfont = ('Helvetica 15')
    def b_sug():
        return sg.Text(" ", font=('Helvetica 15'))
    suggested_text = b_sug()

    def b_in():
        return sg.Input(font=inputfont)
    
    input_field = b_in()
    layout = [
        [suggested_text],
        [b1(), b1b()],
        [b2(), b2b()],
        [b3(), b3b()],
        [input_field],
        [b4(), b5(), b6(), b7()]
    ]
    
    form.Layout(layout)
    while keep_going:
        real_nl_cmd = read_suggest_file()
        print("Suggested CMD", real_nl_cmd)
        
        def b_lay():
            return [sg.Text("    Down:  Use suggested: " + real_nl_cmd, font=('Helvetica 15'))]
        layout[0] = b_lay()

        #suggested_text.Update()

        # ---===--- Loop taking in user input --- #
        while True:
            # button, values = form.ReadNonBlocking()
            button, values = form.Read()
            # print("Got: ", button, values)
            print("button:")
            print(repr(button))
            print("values:")
            print(repr(values))
            nl_command = values[0] if values else ""

            if button in ["Next", "Right:114"]:
                nl_command = "CMD: Next"
                input_field.Update("")
                break

            elif button in ["OK", "\r"] and nl_command:
                print("Executing: " + str(nl_command))
                break

            elif button in ["Reset", "Left:113"] or (button and ord(button[0]) == 0x001B):
                print("Reseting")
                nl_command = "CMD: Reset"
                #input_field.Update("")
                break

            elif button in ["Clear Text"]:
                print("Clearing field")
                input_field.Update("")

            elif button in ["Down:116"]:
                input_field.Update(real_nl_cmd)

            elif button is None and values is None:
                keep_going = False
                break
            time.sleep(0.05)

        write_command_file(nl_command)
