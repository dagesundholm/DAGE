#----------------------------------------------------------------------------------#
#    Copyright (c) 2010-2018 Pauli Parkkinen, Eelis Solala, Wen-Hua Xu,            #
#                            Sergio Losilla, Elias Toivanen, Jonas Juselius        #
#                                                                                  #
#    Permission is hereby granted, free of charge, to any person obtaining a copy  #
#    of this software and associated documentation files (the "Software"), to deal #
#    in the Software without restriction, including without limitation the rights  #
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     #
#    copies of the Software, and to permit persons to whom the Software is         #
#    furnished to do so, subject to the following conditions:                      #
#                                                                                  #
#    The above copyright notice and this permission notice shall be included in all#
#    copies or substantial portions of the Software.                               #
#                                                                                  #
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    #
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      #
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   #
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        #
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, #
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE #
#    SOFTWARE.                                                                     #
#----------------------------------------------------------------------------------# 
get_filename_component(PYTHON_NAME ${PYTHON_EXECUTABLE} NAME)
install(FILES generate_objects.py input_reader.py integration_test.py input_parameters.xml __init__.py DESTINATION
    lib/${PYTHON_NAME}/site-packages/dage COMPONENT dage)
install(DIRECTORY integration_test_files DESTINATION
    lib/${PYTHON_NAME}/site-packages/dage COMPONENT dage)
install (FILES dage.x  COMPONENT dage DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_EXECUTE WORLD_READ WORLD_WRITE )
