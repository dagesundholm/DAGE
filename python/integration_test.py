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


import dage.dage_fortran
from dage.input_reader import InputXML, SCFEnergeticsXML, ActionXML, StructureXML
from collections import OrderedDict
import sys
import os
import shutil
#import dage.dage_fortran

class InvalidTestFileException(Exception):
    pass

fortran_interface_tests = {"io_test":\
            {"name": "Input/Ouput Test",\
             "fail_message": "Input XML and output XML from Fortran are interpreted differently.",\
             "filename": "{}/integration_test_files/input_files/io_test_input.xml".format(os.getcwd())},
         "scf_test":\
            {"name": "SCF Test", \
             "fail_message": "Optimizing electron structure resulted in an unexpected result.",\
             "filename": "{}/integration_test_files/input_files/scf_test_input.xml".format(os.getcwd())},
         "scf_gga_test":\
            {"name": "SCF Test", \
             "fail_message": "Optimizing electron structure resulted in an unexpected result.",\
             "filename": "{}/integration_test_files/input_files/scf_gga_test_input.xml".format(os.getcwd())},
         "scf_atoms_test":\
            {"name": "SCF Test", \
             "fail_message": "Optimizing electron structure resulted in an unexpected result.",\
             "filename": "{}/integration_test_files/input_files/scf_atoms_test_input.xml".format(os.getcwd())},
         "scf_test2":\
            {"name": "SCF Test 2", \
             "fail_message": "Optimizing electron structure resulted in an unexpected result.",\
             "filename": "{}/integration_test_files/input_files/scf_test2_input.xml".format(os.getcwd())}, 
         "ci_test":\
            {"name": "CI Test 2", \
             "fail_message": "Optimizing electron structure resulted in an unexpected result.",\
             "filename": "{}/integration_test_files/input_files/ci_test_input.xml".format(os.getcwd())}}

fortran_interface_subtests = {\
       "structure_test": \
           {"name": "Structure test",\
            "fail_message": "Output structure was not equal to the comparison structure."},\
       "energetics_test": \
           {"name": "Energetics test",\
            "fail_message": "Output energetics were not equal with the comparison energetics."}}
 
def fortran_interface_test(fortran_test_id):
    """
        Performs a test that uses the fortran interface to perform it.
        These kinds of test rely on comparing a comparison output file 
        with the output file from the fortran interface.
    """
    # init the variable for the results of the test
    test_results = {}
    
    # perform the calculations in the test file
    print("filename", fortran_interface_tests[fortran_test_id]["filename"])
    input_xml = InputXML(filename = fortran_interface_tests[fortran_test_id]["filename"])
    kwargs = input_xml.prepare()
    dage.dage_fortran.python_interface.run(**kwargs)
    
    # go through the children of the input file to find the actions
    actions = []
    for child in input_xml.children:
        if isinstance(child, ActionXML):
            actions.append(child)
           
    scf_energetics_definition = input_xml.get_definition_tag("scf_energetics_input")
    test_result = True
    # go through the actions to find the output folders and corresponding SCFEnergetics
    # and possibly structures
    for action in actions:
        if 'output_folder' in action.parameter_values:
            output_folder = action.parameter_values['output_folder']
            test_results[output_folder] = {}
            
            scf_energetics_filename = \
                os.path.join(action.parameter_values['output_folder'], "scf_energetics.xml")
            comparison_scf_energetics_filename = \
                os.path.join(action.parameter_values['output_folder'], "scf_energetics.comparison.xml")
           
            if not os.path.exists(scf_energetics_filename):
                raise Exception("The path for scf-energetics does not exist: '{}'".format(scf_energetics_filename))
            if not os.path.exists(comparison_scf_energetics_filename):
                raise InvalidTestFileException("The path for comparison scf-energetics does not exist: '{}'".format(comparison_scf_energetics_filename))
           
            # if both scf energetics files exist, parse them and compare with each other
            scf_energetics_definition = input_xml.definition.find('scf_energetics_input')
            scf_energetics = SCFEnergeticsXML(definition = scf_energetics_definition)
            scf_energetics.root = scf_energetics.retrieve_path(scf_energetics_filename, None)
            scf_energetics.parse()
            
            comparison_scf_energetics = SCFEnergeticsXML(definition = scf_energetics_definition)
            comparison_scf_energetics.root = comparison_scf_energetics.retrieve_path(comparison_scf_energetics_filename, None)
            comparison_scf_energetics.parse()
            
            # check if energetics are equal, if not fail the test
            if comparison_scf_energetics != scf_energetics:
                test_results[output_folder]["energetics_test"] = False
                test_result = False
            else:
                test_results[output_folder]["energetics_test"] = True
                
            
            structure_filename = \
                os.path.join(action.parameter_values['output_folder'], "structure.xml")
            
            # if structure file exists, parse it and add it as a child of the root 
            # and set it as the input structure of the action
            if os.path.exists(structure_filename):
                structure_definition = input_xml.definition.find('structure_input')
                
                comparison_structure_filename = \
                    os.path.join(action.parameter_values['output_folder'], "structure.comparison.xml")
                
                if not os.path.exists(comparison_structure_filename):
                    raise InvalidTestFileException("The comparison filename for structure does not exist at path: '{}'."\
                                    .format(comparison_structure_filename))
                
                # load the structure
                structure = StructureXML(definition = structure_definition)
                structure.root = structure.retrieve_path(structure_filename, None)
                structure.parse()
                
                # load the comparison structure
                comparison_structure = StructureXML(definition = structure_definition)
                comparison_structure.root = comparison_structure.retrieve_path(comparison_structure_filename, None)
                comparison_structure.parse()
                
                # check if structures are equal, if not fail the test
                if comparison_structure != structure:
                    test_results[output_folder]["structure_test"] = False
                    test_result = False
                else:
                    test_results[output_folder]["structure_test"] = True
    
    # print out the results
    if test_result:
        print("{} succeeded!".format(fortran_interface_tests[fortran_test_id]['name']))
    else:
        print("{} failed!".format(fortran_interface_tests[fortran_test_id]['name']))
        for folder in test_results:
            for test_id in test_results[folder]:
                if not test_results[folder][test_id]:
                    print("{} failed for folder '{}'. Reason: {}" \
                        .format(fortran_interface_subtests[test_id]['name'], \
                                folder, \
                                fortran_interface_subtests[test_id]['fail_message']))
    return test_result

def run_fortran_interface_tests():
    for test_id in fortran_interface_tests:
        try:
            fortran_interface_test(test_id)
        except InvalidTestFileException as e:
            print(str(e))
            
def run_tests(test_name):
    #try:
    source_dir      = os.path.join(os.path.dirname(os.path.realpath(__file__)), "integration_test_files/")
    destination_dir = os.path.join(os.getcwd(), "integration_test_files/")
    if os.path.exists(destination_dir):
        print("integration_test_files folder already exists! This could cause false negative results,"\
              +"if the tests have been altered after the creation of the folder.")
    else:
        print("copying tests from {} to {}".format(source_dir, destination_dir))
        shutil.copytree(source_dir, destination_dir)
    #except:
    #    print "Failed to copy the integration test files to current directory. "+\
    #          "Make sure you are running the tests at a directory you have access rights to"
    #    return 
    if test_name == "integration_test":
        run_fortran_interface_tests()
    else:
        try:
            fortran_interface_test(test_name)
        except InvalidTestFileException as e:
            print(str(e))
    
    
#except:
#    print "Unexpected error occurred: {}".format(str(e))

