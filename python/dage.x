#!/usr/bin/env python
"""---------------------------------------------------------------------------------*
 *    Copyright (c) 2010-2018 Pauli Parkkinen, Eelis Solala, Wen-Hua Xu,            *
 *                            Sergio Losilla, Elias Toivanen, Jonas Juselius        *
 *                                                                                  *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy  *
 *    of this software and associated documentation files (the "Software"), to deal *
 *    in the Software without restriction, including without limitation the rights  *
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     *
 *    copies of the Software, and to permit persons to whom the Software is         *
 *    furnished to do so, subject to the following conditions:                      *
 *                                                                                  *
 *    The above copyright notice and this permission notice shall be included in all*
 *    copies or substantial portions of the Software.                               *
 *                                                                                  *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   *
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, *
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE *
 *    SOFTWARE.                                                                     *
 *----------------------------------------------------------------------------------"""
if __name__ == "__main__":
    from dage.input_reader import InputXML
    from collections import OrderedDict
    import sys
    import os
    from dage.integration_test import run_tests, fortran_interface_tests
    
    error_message = "Give the input file name or command as an input. Valid commands are 'integration_test'"
    for key in list(fortran_interface_tests.keys()):
        error_message += ", {}".format(key)
    
    if len(sys.argv) <= 1:
        print(error_message)
    else:
        if sys.argv[1].endswith(".xml"):
            inp = InputXML(filename = sys.argv[1])
            kwargs = inp.prepare()
            if len(sys.argv) == 2 or sys.argv[2] != "test":
                import dage.dage_fortran
                dage.dage_fortran.python_interface.run(**kwargs)
        elif sys.argv[1] == "integration_test" or sys.argv[1] in list(fortran_interface_tests.keys()):
            run_tests(sys.argv[1])
        else:
            print(error_message)
