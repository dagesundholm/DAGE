
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
 # Input file reader
import os
import sys
import xml.etree.ElementTree as ET
import numpy, ast
from generate_objects import SettingsGenerator
from collections import OrderedDict

class InputProgrammingError(Exception):
    pass

class InputXML(object):
    tag_type = 'input'
    definition_tag = 'input_definition'
    def __init__(self, filename = None, \
                       definition_filename = None,\
                       input_object = None,\
                       parent_object = None,\
                       definition = None, \
                       directory = None):
        
        
        if (input_object is not None):
            self.root = input_object
        elif filename is not None:
            if definition_filename is None:
                definition_filename = os.path.dirname(os.path.realpath(__file__))+"/input_parameters.xml"
            if os.path.exists(filename):
                self.tree = ET.parse(filename)
                self.root = self.tree.getroot()
            else:
                print "Path for definition file: '{}' does not exist".format(filename)
        else:
            self.root = None
        
        self.parent_object = parent_object
        if directory is not None:
            self.directory = directory
        elif filename is not None and os.path.exists(filename):
            self.directory = os.path.dirname(filename)
        elif self.parent_object is not None:
            self.directory = self.parent_object.directory
        else:
            self.directory = None
            
        if definition is not None:
            self.definition = definition
        elif definition_filename is not None:
            if os.path.exists(definition_filename):
                definition = ET.parse(definition_filename)
                self.definition = definition.getroot()
            else:
                sys.exit("Input definition filename does not exist: {}".format(definition_filename))
        elif self.parent_object is not None:
            definition = self.parent_object.definition.find(self.definition_tag)
            if definition is not None:
                self.definition = definition
            else:
                sys.exit("Definition tag '{}' not found from parent definition tree", self.definition_tag)
        else:
            sys.exit("Definition tag input not given.")
        
        self.retrieve()
    
    def prepare(self):
        """
            Prepare the input to have all things required to 
            call the Fortran interface
        """
        self.parse()
        self.handle_folders()
        self.fill_id_values()
        kwargs = OrderedDict()
        self.get_interface_argument_values(kwargs)
        return kwargs
    
    def form_new_directory_path(self, path_text, original_directory = None):
        """
            Creates a new directory path from 'path_text' and 'original_directory' and 
            validate that it exists. Returns the new path.
        """
        if original_directory is not None:
            complete_path = os.path.join(original_directory, path_text)
        else:
            complete_path = path_text
            
        directory_path = os.path.dirname(complete_path)
            
        # check if the path exists
        if not os.path.exists(directory_path):
            raise Exception("Error: '{}' tag path '{}' does not exist".format(self.tag_type, complete_path))
        return directory_path
        
    
    def retrieve_path(self, path_text, directory):
        """ 
            Retrieves content of xml file at path 'path_text'
            to and store it to 'parameter_name' atribute of 'self'.
        """
        
        if directory is not None:
            complete_path = os.path.join(directory, path_text)
        else:
            complete_path = path_text
            
        # check if the path exists
        if os.path.exists(complete_path):
            tree = ET.parse(complete_path)
            return tree.getroot()
        else:
            raise Exception("Error: '{}' tag path '{}' does not exist".format(self.tag_type, complete_path))
        
    def retrieve(self):
        """ 
            Retrieves content to the tag from external file(s),
            if the tag has attribute or child named 'path' and/or
            'extends_path'.

        """
        
        
        if self.root is not None:
            
            # check if current tag has an attribute or child with
            # name 'path'
            path_text = InputXML.read_tag_or_attribute_value(self.root, 'path')
                
            # try to retrieve the content from path_text
            if path_text is not None and path_text != "":
                try:
                    self.root      = self.retrieve_path(path_text, self.directory)
                    self.directory = self.form_new_directory_path(path_text, self.directory)
                except Exception, e:
                    sys.exit(str(e))
            
            # check if current tag has an attribute or child with
            # name 'extends_path'
            path_text = InputXML.read_tag_or_attribute_value(self.root, 'extends_path')
            self.extends_roots = []
            self.extends_directories = []
            directory = self.directory
        
            while path_text is not None:
                # try to retrieve the content from path_text
                try:
                    self.extends_roots.append(self.retrieve_path(path_text, directory))
                    self.extends_directories.append(self.form_new_directory_path(path_text, directory))
                except Exception, e:
                    sys.exit(str(e))
                # prepare for the next loop by getting the next extends path and corresponding directory
                directory = self.extends_directories[-1]
                path_text = InputXML.read_tag_or_attribute_value(self.extends_roots[-1], 'extends_path')
                  
    def fill_id_values(self):
        """
            Finds the id for each parameter where reference is made with name
            and fills it to the correct place
        """
        for parameter_name in self.parameter_values:
            if parameter_name.endswith("_id"):
                # check if the tag has value that is not 0, in that case 
                # we are not finding the value
                if self.get_parameter_value(parameter_name) == 0:
                    tagtype = parameter_name[:parameter_name.rfind('_')]
                    name_tag_found = tagtype+"_name" in self.parameter_values
                    if name_tag_found:
                        name = self.parameter_values[tagtype+"_name"]
                        if name is not None and name != "":
                            id_value = self.get_tagid_for_name(tagtype, name)
                            if id_value != -1:
                                self.parameter_values[parameter_name] = id_value
                    
        for child in self.children:
            child.fill_id_values()
                    
    def get_tagid_for_name(self, tagtype, name):
        if self.parent_object is not None:
            for child in self.parent_object.children:
                if hasattr(child, 'tag_type') and child.tag_type == tagtype and hasattr(child, 'name') and child.name == name:
                    return child.id
        return -1
    
    def get_parameter_definition(self, parameter_name):
        """
            Retrieve the parameter definition for parameter name 
            'parameter_name'.
        """
        for parameter_definition in self.definition.findall('parameter'):
            if parameter_definition.attrib['name'] == parameter_name:
                return parameter_definition
        return None
    
    def get_definition_tag(self, tag_name):
        """
            Retrieve the definition tag for a tag with name = tag_name
        """
        definition = self.definition.find('{}'.format(tag_name)) 
        return definition
    
    
    def _parse_children(self, root, directory):
        """
            Parse children of root xml-tag 'root' and store them as
            children in the 'self'.
            
            Note: this function is a subfunctionality of function 'parse'
                  and it should not be used independently.
        """
        for tag in root:
            if tag.tag not in self.parameter_values:
                # try to find the correct definition tag by using the "*_input"-format 
                definition = self.definition.find('{}_input'.format(tag.tag)) 
                    
                # if the input definition was not found, try to find the definition from 
                # the '<class>'-tags
                if definition is None:
                    definition_found = False
                    for definition_tag in self.definition.findall('class'):
                        if definition_tag.attrib['name'] == tag.tag:
                            definition = definition_tag
                            definition_found = True
                            break
                            
                    if not definition_found:
                        print "Warning: Found unknown tag with name '{}'. Ignoring.".format(tag.tag)
                        continue
                    else:
                        child = InputXML(parent_object = self, definition = definition, input_object = tag, directory = directory)
                            
                else:
                    if tag.tag == 'settings':
                        child = SettingsXML(parent_object = self, definition = definition, input_object = tag, directory = directory)
                    elif tag.tag == 'structure':
                        child = StructureXML(parent_object = self, definition = definition, input_object = tag, directory = directory)
                    elif tag.tag == 'basis_set':
                        child = BasisSetXML(parent_object = self, definition = definition, input_object = tag, directory = directory)
                    elif tag.tag == 'action':
                        child = ActionXML(parent_object = self, definition = definition, input_object = tag, directory = directory)
                    elif tag.tag == 'scf_energetics':
                        child = SCFEnergeticsXML(parent_object = self, definition = definition, input_object = tag, directory = directory)
                            
                self.children.append(child)
                self.child_definitions.append(tag.tag)
                self.add_counters(child)
                child.parse()
     
    def parse(self):
        """
            Parse paremeters and child xml-tags of the root-xml tags stored 
            in self.root and self.extends_roots. Stores the found child-xml classes 
            to 'self.children' and the parameter values to 'self.parameter_values'.
            The corresponding definitions are stored to 'self.child_definitions' and 
            'self.parameter_definitions', respectively.
            
            User must note that this function is recursive as it calls 'parse' for
            all found children in '_parse_children' calls.
        """
        self.parameter_values = OrderedDict()
        self.parameter_definitions = OrderedDict()
        self.children = []
        self.child_definitions = []
        # handle the parameters first
        for parameter_definition in self.definition.findall('parameter'):
            
            if SettingsGenerator.is_valid_parameter(parameter_definition):
                self.set_parameter_value(parameter_definition, self.read_parameter_value(parameter_definition))
                self.parameter_definitions[parameter_definition.attrib['name']] = parameter_definition
                
                if parameter_definition.attrib['name'] == 'name':
                    self.name = self.parameter_values['name']
            else:
                print "PARAMETER is not valid", parameter_definition.attrib['name']
                
        # if the object has extends_root, then parse the children from it
        # and store them to 'self'
        if      hasattr(self, 'extends_roots')       and self.extends_roots is not None\
            and hasattr(self, 'extends_directories') and self.extends_directories is not None: 
                
            for i, extends_root in enumerate(self.extends_roots):
                self._parse_children(extends_root, self.extends_directories[i])
        
        # parse the children from the xml-root of this object and store them 
        # to 'self'
        if self.root is not None:
            self._parse_children(self.root, self.directory)
                   
        # add the tag classes that are not found in the input file, just to 
        # input the default values.
        for definition_tag in self.definition.findall('class'):
            if definition_tag.attrib['name'] not in self.child_definitions:
                child = InputXML(parent_object = self, definition = definition_tag)
                self.children.append(child)
                child.parse()
              
     
    def handle_folders(self):
        """
           Creates missing folders and replaces relative paths with
           non-relative ones
        """
        for parameter_name in self.parameter_values:
            if parameter_name in ['output_folder', 'input_folder', 'folder_path']:
                if self.parameter_values[parameter_name] is not None:
                    # convert the non absolute paths to absolute ones
                    if not os.path.isabs(self.parameter_values[parameter_name]):
                        # join the directory of the file with the input directory
                        path = os.path.join(self.directory, self.parameter_values[parameter_name])
                        
                        # make the path more readable by removing extra slashes and dots
                        self.parameter_values[parameter_name] = os.path.normpath(path)
                        
                    
                    # if the output folder does not exist, create it
                    if parameter_name == 'output_folder' and not os.path.exists(self.parameter_values[parameter_name]):
                        os.makedirs(self.parameter_values[parameter_name])
             
        for child in self.children:
            child.handle_folders()
                
            
     
    def get_interface_argument_values(self, argument_values, parameter_definitions = {}, abbreviation = None, counter_present = False):
        """
            This function converts the values of the parameters to a form suitable for the
            Fortran interface. The converted values are stored to input-output dictionary 'arguments_values'.
            
        """
        if 'abbreviation' in self.definition.attrib:
            abbreviation = self.definition.attrib['abbreviation']
        for parameter_name in self.parameter_values:
            if SettingsGenerator.generate_fortran(self.parameter_definitions[parameter_name]):
                if abbreviation is not None:
                    argument_key = "{}_{}".format(abbreviation, parameter_name)
                else:
                    argument_key = parameter_name
                if counter_present:
                    # Check if the parameter value is None. If the value is None, the 
                    # parameter is not present in the input file, and the default 
                    # value of the parameter is not specified.
                    if self.parameter_values[parameter_name] is not None:
                        if argument_key in argument_values and argument_values[argument_key] is not None: 
                            argument_values[argument_key].append(self.parameter_values[parameter_name])
                        else:
                            argument_values[argument_key] = [self.parameter_values[parameter_name]]
                            parameter_definitions[argument_key] = self.parameter_definitions[parameter_name]
                    else:
                        if argument_key not in parameter_definitions:
                            argument_values[argument_key] = None
                            parameter_definitions[argument_key] = self.parameter_definitions[parameter_name]
                else: 
                    if argument_key in argument_values:
                        print "Warning: Found two (or more) arguments for the same parameter: {}".format(argument_key)
                    else:
                        argument_values[argument_key] = self.parameter_values[parameter_name]
                        parameter_definitions[argument_key] = self.parameter_definitions[parameter_name]
             
        for child in self.children:
            if 'global_index_counter' in child.definition.attrib or 'local_index_counter' in child.definition.attrib or 'counters' in child.definition.attrib:
                counter_present = True
            if SettingsGenerator.generate_fortran(child.definition):
                child.get_interface_argument_values(argument_values, parameter_definitions, abbreviation = abbreviation, counter_present = counter_present)
         
        # if we are at the root, convert the values with type list to numpy arrays
        if self.parent_object is None:
            for argument_key in argument_values:
                # the string lists need some special attention:
                if parameter_definitions[argument_key].attrib['type'].startswith('string') and type(argument_values[argument_key]) == list:
                    temp = numpy.empty((256, len(argument_values[argument_key])+1), dtype="c")
                    for j, value in enumerate(argument_values[argument_key]):
                        temp[:, j] = "{0:{width}}".format(argument_values[argument_key][j], width=256)
                    argument_values[argument_key] = numpy.array(temp, dtype="c").T
                elif type(argument_values[argument_key]) == list:
                    temp_array = numpy.array(argument_values[argument_key], order='F').T
                    shape = temp_array.shape
                    if len(shape) == 3:
                        new_shape = (shape[0], shape[1], shape[2]+1)
                    elif len(shape) == 2:
                        new_shape = (shape[0], shape[1]+1)
                    else:
                        new_shape = (shape[0]+1)
                        
                    new_array = numpy.empty(new_shape, order='F')
                    if len(shape) == 3:
                        new_array[:, :, :shape[2]] = temp_array[:, :, :]
                    elif len(shape) == 2:
                        new_array[:, :shape[1]] = temp_array[:, :]
                    else:
                        new_array[:shape[0]] = temp_array[:]
                    
                    argument_values[argument_key] = new_array
                elif argument_values[argument_key] is None:
                    del argument_values[argument_key]
                
    def add_counters(self, child):
        """
            Add all the counter values for the child object 'child' of 'self' by one
        """
        if 'global_index_counter' in child.definition.attrib:
            success = self.add_counter_value(child.definition.attrib['global_index_counter'])
            if not success:
                print "Warning: Adding counter {} failed. Counter not found.".format(child.definition.attrib['global_index_counter'])
            else:
                child.id = self.get_counter_value(child.definition.attrib['global_index_counter'])
            if 'local_index_counter' in child.definition.attrib:
                success = self.add_counter_value(child.definition.attrib['local_index_counter'])
                if not success:
                    print "Warning: Adding counter {} failed. Counter not found.".format(child.definition.attrib['local_index_counter'])
            if 'counters' in child.definition.attrib:
                success = self.add_counter_value(child.definition.attrib['counters'])
                if not success:
                    print "Warning: Adding counter {} failed. Counter not found.".format(child.definition.attrib['counters'])
    
    def add_counter_value(self, counter_name):
        """
            Add value of counter parameter with name=='counter_name' by one.
            If the counter is not found in the local object, it
            is seached from the parent objects.
        """
        if counter_name in self.parameter_values:
            if self.parameter_values[counter_name] is None:
                self.parameter_values[counter_name] = 0
            self.parameter_values[counter_name] += 1
            return True
        else:
            if self.parent_object is not None:
                return self.parent_object.add_counter_value(counter_name)
            else:
                return False
            
    def get_counter_value(self, counter_name):
        """ 
            Get the value of a counter with name 'counter_name'.
            If the counter is not found in the local object, it
            is seached from the parent objects.
        """
        if counter_name in self.parameter_values:
            return self.parameter_values[counter_name]
        else:
            if self.parent_object is not None:
                return self.parent_object.get_counter_value(counter_name)
            else:
                return -1
        
   
    def set_parameter_value(self, parameter_definition, value):
        """
            Set an arbitrary value 'value' for the parameter with definition
            'parameter_definition'.
        """
      
        # convert the value to right data type and check that it is valid
        final_value = self.convert_argument_value(value, parameter_definition)
        
        # check that value is within given limits
        self.check_value_range(final_value, parameter_definition)
        
        # set the parameter value
        self.parameter_values[parameter_definition.attrib['name']] = final_value
        
    @staticmethod
    def read_tag_or_attribute_value(root, name):
        """
            Reads the value of a tag or attribute with name 'name' in an xml. If 
            attribute or tag is not found, None is returned. 
        """
        value = None
        if root is not None:
            tag = root.find(name)
            if tag is not None:
                value = tag.text
            elif name in root.attrib:
                value = root.attrib[name]
        return value
   
    def read_parameter_value(self, parameter_definition):
        """ 
            Read the value of the parameter first from the values of the XML-element, 
            secondarily from the objects we are extending from and thirdly from 
            the default value of the parameter definition.
        """
        value = InputXML.read_tag_or_attribute_value(self.root, parameter_definition.attrib['name'])
        
        # if value is not found at root, then use the value from extends roots
        if value is None and hasattr(self, 'extends_roots')       and self.extends_roots is not None: 
            for extends_root in self.extends_roots:
                value = InputXML.read_tag_or_attribute_value(extends_root, parameter_definition.attrib['name'])
                
                # if value is found, break the iteration
                if value is not None:
                    break
                
        # fall back to default value/or None if one is not specified
        if value is None:
            if 'default' in parameter_definition.attrib:
                value = parameter_definition.attrib['default']
                
        return value
    
    def get_parameter_value(self, parameter_name):
        """ 
            Get the value of the parameter from the parsed parameters.
            If the parameter is not found an InputProgrammingError
            is raised.
        """
        if hasattr(self, 'parameter_values') and parameter_name in self.parameter_values:
            return self.parameter_values[parameter_name]
        else:
            raise InputProgrammingError("Accessed parameter: '{}' is not in the values ".format(parameter_name)+ \
                                        "of the object. Have you perfomed 'parse' for the object?")
        
    def parameter_values_are_equal(self, other, parameter_name):
        """
            Compare the values of parameter with name 'parameter_name' for
            two objects of the same type.
        """
        # check that the input objects are of same type
        if type(self) != type(other):
            raise InputProgrammingError("The objects compared with parameter_values_are_equal"+
                                        " are not of same type.") 
        
        # get the values for both input objects
        self_value = self.get_parameter_value(parameter_name)
        other_value = other.get_parameter_value(parameter_name)
        
        if isinstance(self_value, list) or isinstance(self_value, numpy.ndarray):
            if len(self_value) != len(other_value):
                return False
            for i in range(len(self_value)):
                if type(self_value[i]) == float or type(self_value[i]) == numpy.float64  or type(self_value[i]) == numpy.float32  or type(self_value[i]) == numpy.float16:
                    if abs(self_value[i] - other_value[i]) > 1e-10:
                        return False
                elif self_value[i] != other_value[i]:
                    return False
            return True
        else:
            return self_value == other_value
    
    def all_parameter_values_are_equal(self, other):
        """
            Check if all parameter values of 'self' and 'other'
            are equal
        """
        for parameter_name in self.parameter_values:
            if not self.parameter_values_are_equal(other, parameter_name):
                return False
        return True
    
    def is_of_same_type_as(self, other):
        """
            Check if self is of same type as other
        """
        return     type(self) == type(other) \
               and self.definition.attrib['name'] == other.definition.attrib['name']
    
    
    def children_are_equal(self, other):
        """
            Check if children of 'self' and 'other' are equal with definition 
            and value
        """
        for child in self.children:
            equal_found = False
            # go through all the children and check if there is equal
            for other_child in other.children:
                if child == other_child:
                    equal_found = True
                    
            # if not, the children cannot be equal
            if not equal_found:
                return False
        return True

    def __eq__(self, other):
        """
            Check if two InputXML objects are equal with each other
        """
        return     self.is_of_same_type_as(other)\
               and self.all_parameter_values_are_equal(other)\
               and self.children_are_equal(other) 
           
    
    def __ne__(self, other):
        return not self.__eq__(other)
        
        
    
            
                
    def read_array_values(self, value_text, argument_type):
        is_number = argument_type.startswith("int") or \
                    argument_type.startswith("float") or \
                    argument_type.startswith("double")
        # try to evaluate the molecular orbitals as dict 
        try:
            dictionary = ast.literal_eval("{"+ value_text +"}")
            size = max(dictionary.keys())
                
            # init array of size
            if is_number:
                result = [0] * size
            else:
                result = [None] * size
            for key in dictionary:
                # convert the indexing from the 1-starting to 0-starting
                result[key-1] = dictionary[key]
        except:
            try:
                result = ast.literal_eval("["+ value_text +"]")
            except:
                raise Exception("Bad form of array, should have a list or a dictionary, value is: {}.".format(value_text))  
        return result
                
    def convert_argument_value(self, value_text, parameter_definition):
        argument_type = parameter_definition.attrib['type']
        if SettingsGenerator.has_options(parameter_definition):
            value_text = self.get_option_value(value_text, parameter_definition)

        if SettingsGenerator.is_array(parameter_definition):
            if value_text is None:
                value = None
            else:
                # do the parsing of the input array (could also be a dictionary), which
                # has to be changed to a list
                array_values = self.read_array_values(value_text, argument_type)
                
                # get the final size of the result array from the parameter definition
                size = int(parameter_definition.attrib['shape'])
                value = numpy.zeros(size)
                try:
                    for i, arg in enumerate(array_values):
                        if argument_type.startswith('int'):
                            value[i] = int(arg)
                        if argument_type.startswith('float'):
                            value[i] = float(arg)
                        if argument_type.startswith('double'):
                            value[i] = float(arg)
                        if argument_type.startswith('string'):
                            if SettingsGenerator.generate_fortran(parameter_definition):
                                value[i] = str(arg)
                            else:
                                value[i] = str(arg)
                        if argument_type.startswith('bool'):
                            if arg.lower() == 'false':
                                value[i] = False
                            elif arg.lower() == 'true':
                                value[i] = True
                            else:
                                value[i] = bool(arg)
                except ValueError:
                    sys.exit('Error: parameter with type \'{}\' and name \'{}\' has invalid value: \'{}\''.format(argument_type, parameter_definition.attrib['name'], value_text))
        else:
            try:
                if value_text is None:
                    value = None
                elif argument_type.startswith('int'):
                    value = int(value_text)
                elif argument_type.startswith('float'):
                    value = float(value_text)
                elif argument_type.startswith('double'):
                    value = float(value_text)
                elif argument_type.startswith('string'):
                    if SettingsGenerator.generate_fortran(parameter_definition):
                        value = str(value_text)
                    else:
                        value = str(value_text)
                elif argument_type.startswith('bool'):
                    if value_text.lower() == 'false':
                        value = False
                    elif value_text.lower() == 'true':
                        value = True
                    else:
                        value = bool(arg)
            except ValueError:
                sys.exit('Error: parameter with type \'{}\' and name \'{}\' has invalid value: \'{}\''.format(argument_type, parameter_definition.attrib['name'], value_text))
        return value
    
    def check_value_range(self, value, parameter_definition):
        if value is not None:
            if 'minval' in parameter_definition.attrib:
                minval = parameter_definition.attrib['minval']
                if value < float(minval):
                    sys.exit('Error: argument with name {} and value {} is smaller than the smallest allowed value: {}', parameter_definition.attrib['name'], value, float(minval))
            if 'maxval' in parameter_definition.attrib:
                maxval = parameter_definition.attrib['maxval']
                if value > float(maxval):
                    sys.exit('Error: argument with name {} and value {} is larger than the largest allowed value: {}', parameter_definition.attrib['name'], value, float(maxval))
    
    def get_option_value(self, value_text, parameter_definition):
        options = parameter_definition.findall('option')
        result = None
        if len(options) > 0:
            valid_options = ""
            for option in options:
                if 'value' in option.attrib and value_text == option.attrib['value']:
                    return value_text
                elif 'text_value' in option.attrib and value_text == option.attrib['text_value']:
                    return option.attrib['value']
                else:
                    valid_options += ("{}: {} ".format(option.attrib['value'], option.attrib['text_value']))
            sys.exit('Error: The value "{}" for argument with name "{}" is not within allowed options: {} '.format(value_text, parameter_definition.attrib['name'], valid_options))
    
    
    def get_root_object(self):
        if self.parent_object is None:
            return self
        else:
            return self.parent_object.get_root_object()
        
    
class SCFEnergeticsXML(InputXML):
    tag_type = 'scf_energetics'
    definition_tag = 'scf_energetics_input'
        
            
class ActionXML(InputXML):
    tag_type = 'action'
    definition_tag = 'action_input'
    
    def parse(self):
        super(ActionXML, self).parse()
        self.handle_output_files()
        
    
    def handle_output_files(self):
        """
            Reads in the output files and creates the corresponding 
            objects to the tree
        """
        if 'output_folder' in self.parameter_values:
            scf_energetics_filename = \
                os.path.join(self.parameter_values['output_folder'], "scf_energetics.xml")
            root_object = self.get_root_object()
            
            # if scf energetics file exists, parse it and add as a child of the root 
            # and set it as the input scf energetics of the action
            if os.path.exists(os.path.join(self.directory, scf_energetics_filename)):
                scf_energetics_definition = root_object.definition.find('scf_energetics_input')
                scf_energetics = SCFEnergeticsXML(parent_object = root_object, \
                                                  definition = scf_energetics_definition)
                scf_energetics.root = scf_energetics.retrieve_path(scf_energetics_filename, scf_energetics.directory)
                root_object.children.append(scf_energetics)
                root_object.child_definitions.append('scf_energetics')
                root_object.add_counters(scf_energetics)
                scf_energetics.parse()
                
                scf_energetics_id_definition = self.get_parameter_definition('scf_energetics_id')
                self.set_parameter_value(scf_energetics_id_definition, scf_energetics.id)
            
            structure_filename = \
                os.path.join(self.parameter_values['output_folder'], "structure.xml")
            
            # if structure file exists, parse it and add it as a child of the root 
            # and set it as the input structure of the action
            if os.path.exists(os.path.join(self.directory, structure_filename)):
                structure_definition = root_object.definition.find('structure_input')
                structure = StructureXML(parent_object = root_object, \
                                                  definition = structure_definition)
                structure.root = structure.retrieve_path(structure_filename, structure.directory)
                root_object.children.append(structure)
                root_object.child_definitions.append('structure')
                root_object.add_counters(structure)
                structure.parse()
                
                structure_id_definition = self.get_parameter_definition('structure_id')
                self.set_parameter_value(structure_id_definition, structure.id)
        
             
            
    
class BasisSetXML(InputXML):
    tag_type = 'basis_set'
    definition_tag = 'basis_set_input'                  
                
        
class SettingsXML(InputXML):
    tag_type = 'settings'
    definition_tag = 'settings_input'
    
    
class StructureXML(InputXML):
    tag_type = 'structure'
    definition_tag = 'structure_input'
    atom_types = {'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10, 'Na': 11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18}
    
    def read_input(self):
        charge = self.root.find('charge')
        # read relative charge 
        if (charge is not None):
            self.charge = int(charge.text)
        else:
            self.charge = 0
            
        # read coordinates and atom types
        self.coordinates = []
        self.types = []
        self.charges = []
        
        # first read atom coordinates in 'atom' tags
        for i, atom in enumerate(self.root.findall('atom')):
            self.read_atom_coordinates_and_type(atom)
     
            
        # then read atoms in 'atoms' tags
        for i, atoms in enumerate(self.root.findall('atoms')):
            self.read_atoms_coordinates_and_types(atoms)
    
    def read_atom_coordinates_and_type(self, atom):
        result = [0.0, 0.0, 0.0]
        x = atom.find('x')
        if (x is not None):
            result[0] = float(x.text)
        y = atom.find('y')
        if (y is not None):
            result[1] = float(y.text)
        z = atom.find('z')
        if (z is not None):
            result[2] = float(z.text)
            
        xyz = atom.find('xyz')
        atom_type = self.read_atom_type(atom) 
        if (xyz is not None):
            xyz_text = xyz.text.strip().split(" ")
            if (len(xyz_text) == 4):
                atom_type = get_atom_type(xyz_text[0])
                atom_charge = get_atom_charge(xyz_text[0])
                result[0] = float(xyz_text[1])
                result[1] = float(xyz_text[2])
                result[2] = float(xyz_text[3])
            else:
                sys.exit("Error: Too many or too few coordinates in 'atom'->'xyz' -tag.")
        
        self.coordinates.append(result)
        self.types.append(atom_type)
        self.charges.append(atom_charge)
        
    def get_atom_type(self, atom_type_text):
        return int(self.atom_types[atom_type_text])
    
    def get_atom_charge(self, atom_type_text):
        return float(self.atom_types[atom_type_text])
    
    def read_atom_type(self, atom):
        if 'type' in atom.attrib:
            return atom.attrib['type']
        else:
            sys.exit("Error: The mandatory attribute 'type' not found in 'atom'-tag")
               
    def read_atoms_coordinates_and_types(self, atoms):
        xyz = atoms.find('xyz')
        coordinates = []
        types = []
        charges = []
        if (xyz is not None):
            xyz_lines = xyz.text.splitlines()
            for xyz in xyz_lines:
                xyz_text = xyz.strip().split(" ")
                xyz_coord = [0.0,  0.0, 0.0]
                # ignore empty lines
                if (len(xyz_text) == 1 and xyz_text[0] == ""):
                    continue
                elif (len(xyz_text) == 4):
                    types.append(self.get_atom_type(xyz_text[0]))
                    charges.append(self.get_atom_charge(xyz_text[0]))
                    xyz_coord[0] = float(xyz_text[1])
                    xyz_coord[1] = float(xyz_text[2])
                    xyz_coord[2] = float(xyz_text[3])
                    coordinates.append(xyz_coord)
                else:
                    sys.exit("Error: Too many or too few coordinates in 'atoms'->'xyz' -line.")
        self.coordinates.extend(coordinates)
        self.types.extend(types)
        self.charges.extend(charges)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print "Give the input file name as an input."
    else:
        inp = InputXML(filename = sys.argv[1], definition_filename = os.path.dirname(os.path.realpath(__file__))+"/input_parameters.xml")
        import dage_fortran
        dage_fortran.python_interface.run(**inp.prepare())
    

    
