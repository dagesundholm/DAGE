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
 # System to generate settings and the interfaces between python and Fortran
import os
import sys
import xml.etree.ElementTree as ET

class InvalidTemplateException(Exception):
    pass

class InputDefinitionException(Exception):
    pass

class SettingsGenerator(object):
    mandatory_tags = ["structure_input", "settings_input",\
        "action_input", "basis_set_input", "scf_energetics_input"]
    def __init__(self, path):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.path = path
        
    def validate_settings_definition(self):
        for mandatory_tag in SettingsGenerator.mandatory_tags:
            found_tags = self.root.findall(mandatory_tag)
            if found_tags is None or len(found_tags) == 0:
                raise InputDefinitionException(\
                    "Mandatory tag '{}' not defined in settings definition at: '{}'."\
                    .format(mandatory_tag, self.path))
            elif len(found_tags) > 1:
                raise InputDefinitionException(\
                    "Mandatory tag '{}' not defined in settings definition at: '{}'."\
                    .format(mandatory_tag, self.path))
    
    def generate_interface(self):
        header_lines = []
        assign_lines = []
        introduction_lines = []
        core_object_name = "core_object"
        header_lines.extend(SettingsGenerator.get_license_lines())
        header_lines.extend(SettingsGenerator.get_auto_generated_lines())
        header_lines.extend(SettingsGenerator.get_divider_lines())
        header_lines.append("module python_interface")
        header_lines.append("    use Core_class")
        header_lines.append("    use LCAO_m")
        header_lines.append("    use ISO_FORTRAN_ENV")
        header_lines.append("    implicit none")
        header_lines.append("contains")
        header_lines.append("    function run( &")
        
        # get all handled settings objects objects
        self.mandatory_objects = []
        for mandatory_tag in SettingsGenerator.mandatory_tags:
            self.mandatory_objects.append(self.root.find(mandatory_tag))
            
        
        # generate header lines of the interface and the introduction lines
        header_lines.extend(self.get_interface_header_lines())
        introduction_lines.extend(self.get_all_interface_parameter_lines())
        
        assign_lines.append("        allocate({}%settings(number_of_settings))".format(core_object_name))
        assign_lines.append("        allocate({}%actions(number_of_actions))".format(core_object_name))
        assign_lines.append("        allocate({}%structures(number_of_structures))".format(core_object_name))
        assign_lines.append("        allocate({}%basis_sets(number_of_basis_sets))".format(core_object_name))
        assign_lines.append("        allocate({}%scf_energetics(number_of_scf_energetics))".format(core_object_name))
            
        for i, mandatory_object in enumerate(self.mandatory_objects):
            assign_lines.extend(SettingsGenerator.get_tag_lines(mandatory_object, core_object_name))
            
        
        introduction_lines.append("        type(Core)                         :: {}".format(core_object_name))
        introduction_lines.append("        logical                            :: success")
        introduction_lines.append("        integer                            :: i, icounter")
        introduction_lines.append("")
        header_lines.extend(introduction_lines)
        header_lines.extend(assign_lines)
        header_lines.append("        success = {}%run()".format(core_object_name))
        header_lines.append("    end function")
        header_lines.append("end module")
        return header_lines
    
    def generate_action(self, template = None):
        header_lines = []
        header_lines.extend(SettingsGenerator.get_license_lines())
        header_lines.extend(SettingsGenerator.get_divider_lines())
        header_lines.extend(SettingsGenerator.get_auto_generated_lines())
        header_lines.extend(SettingsGenerator.get_divider_lines())
        header_lines.append("module Action_class")
        header_lines.append("    use globals_m")
        header_lines.append("    use ISO_FORTRAN_ENV")
        header_lines.append("    implicit none")
        header_lines.append("")
        header_lines.extend(self.generate_input_type("action_input", template))
        header_lines.append("end module")
        return header_lines
    
    def generate_scf_energetics(self, template = None):
        header_lines = []
        header_lines.extend(SettingsGenerator.get_license_lines())
        header_lines.extend(SettingsGenerator.get_divider_lines())
        header_lines.extend(SettingsGenerator.get_auto_generated_lines())
        header_lines.extend(SettingsGenerator.get_divider_lines())
        header_lines.append("module SCFEnergetics_class")
        header_lines.append("    use globals_m")
        header_lines.append("    use ISO_FORTRAN_ENV")
        header_lines.append("    implicit none")
        header_lines.append("")
        header_lines.extend(self.generate_input_type("scf_energetics_input", template))
        if template is not None:
            header_lines.append("contains")
            header_lines.extend(template["content"])
        header_lines.append("end module")
        return header_lines
    
    def generate_settings(self):
        header_lines = []
        header_lines.extend(SettingsGenerator.get_license_lines())
        header_lines.extend(SettingsGenerator.get_auto_generated_lines())
        header_lines.extend(SettingsGenerator.get_divider_lines())
        header_lines.append("module Settings_class")
        header_lines.append("    use globals_m")
        header_lines.append("    use ISO_FORTRAN_ENV")
        header_lines.append("    implicit none")
        header_lines.append("")
        header_lines.extend(self.generate_settings_types())
        header_lines.append("end module")
        return header_lines
        
    def generate_settings_types(self):
        lines = []
        main_type_lines = []
        settings_input = self.root.find("settings_input")
        if settings_input is not None:
            if 'fortran_name' in settings_input.attrib:
                main_class_name = settings_input.attrib['fortran_name']
            else:
                main_class_name = "ProgramSettings"
            lines.append("    public :: {}".format(main_class_name))
            lines.append("")
            for settings_class in settings_input.findall("class"):
                
                option_lines = []
                if SettingsGenerator.generate_fortran(settings_class):
                    
                    if 'fortran_name' in settings_class.attrib:
                        name = settings_class.attrib['fortran_name']
                    elif 'name' in settings_class.attrib:
                        name = settings_class.attrib['name']
                        
                    if 'fortran_parameter_name' in settings_class.attrib:
                        variable_name = settings_class.attrib['fortran_parameter_name']
                    else:
                        variable_name = settings_class.attrib['name']
                    lines.extend(SettingsGenerator.get_comment_lines(settings_class, name, indent="    "))
                    lines.append("    type :: {} ".format(name))
                    main_type_lines.append("        type{0:{width}}::  {1}".format("("+name+")", variable_name, width=28))
                    for parameter in settings_class.findall("parameter"):
                        lines.extend(SettingsGenerator.get_parameter_lines(parameter, is_in_subclass = False))
                        option_lines.extend(SettingsGenerator.get_option_lines(parameter))
                    lines.append("    end type")
                    lines.append("")
                    if len(option_lines) > 0:
                        lines.extend(option_lines)
                        lines.append("")
                    
            
            lines.extend(SettingsGenerator.get_comment_lines(settings_input, name, indent="    "))
            lines.append("    type :: {} ".format(main_class_name))
            lines.extend(main_type_lines)
            lines.append("    end type")

        return lines



    def generate_input_type(self, definition_tag_name, template):
        """ 
            Generates Fortran90 source code lines for an input type.
        """
        lines = []
        option_lines = []
        main_type_lines = []
        input_type = self.root.find(definition_tag_name)
        if input_type is not None:
            main_class_name = input_type.attrib['fortran_name']
            lines.append("    public :: {}".format(main_class_name))
            lines.append("")
            lines.extend(SettingsGenerator.get_comment_lines(input_type, definition_tag_name, "    "))
            lines.append("    type   :: {}".format(main_class_name))
            
            for parameter in input_type.findall("parameter"):
                lines.extend(SettingsGenerator.get_parameter_lines(parameter))
                option_lines.extend(SettingsGenerator.get_option_lines(parameter))
                
            for subclass in input_type.findall("class"):
                if SettingsGenerator.generate_fortran(subclass):
                    for parameter in subclass.findall("parameter"):
                        lines.extend(SettingsGenerator.get_parameter_lines(parameter, is_in_subclass = True))
                        option_lines.extend(SettingsGenerator.get_option_lines(parameter))
                        
            if template is not None and "procedures" in template:
                lines.append("    contains")
                lines.extend(template["procedures"])
            lines.append("")
            lines.append("    end type")
            lines.append("")
            
            # handle the constructors in template
            if template is not None and "constructors" in template:
                lines.extend(template["constructors"])
                lines.append("")
            
            # add option lines
            if len(option_lines) > 0:
                lines.extend(option_lines)
                lines.append("")
              
             
             
        return lines


    def get_all_interface_parameter_lines(self):
        
        tags = [self.root]
        for mandatory_object in self.mandatory_objects:
            tags.append(mandatory_object)
            self.get_sub_classes(mandatory_object, tags)

        lines = []
        abbreviation = None
        counter = None
        for i, tag in enumerate(tags):
            name = None
            if SettingsGenerator.generate_fortran(tag):
                # get values of some key elements of tags
                if 'abbreviation' in tag.attrib:
                    abbreviation = tag.attrib['abbreviation']
                if 'name' in tag.attrib:
                    name = tag.attrib['name']
                if 'global_index_counter' in tag.attrib:
                    counter = tag.attrib['global_index_counter']
                
                # get the parameters
                parameters = tag.findall("parameter")
                
                # and finally get the interface parameter lines
                for j, parameter in enumerate(parameters):
                    if SettingsGenerator.generate_fortran(parameter):
                        lines.extend(SettingsGenerator.get_interface_parameter_lines(parameter, abbreviation, counter))
        
        return lines
            

    def get_interface_header_lines(self):
        
        lines = []
        line = "            "
        line_length = len(line)
        
        tags = [self.root]
        for mandatory_object in self.mandatory_objects:
            tags.append(mandatory_object)
            SettingsGenerator.get_sub_classes(mandatory_object, tags)
            
        abbreviation = None
        for i, tag in enumerate(tags):
            if SettingsGenerator.generate_fortran(tag):
                if 'abbreviation' in tag.attrib:
                    abbreviation = tag.attrib['abbreviation']
                 
                parameters = tag.findall("parameter")
                for j, parameter in enumerate(parameters):
                    if SettingsGenerator.generate_fortran(parameter):
                        if 'name' in parameter.attrib:
                            name = parameter.attrib['name']
                            if abbreviation is not None:
                                parameter_name = "{}_{}".format(abbreviation, name)
                            else:
                                parameter_name = name
                            if j != len(parameters)-1 or i != len(tags)-1:
                                parameter_name += ", "
                                line += parameter_name
                                line_length = len(line)
                                if line_length > 80:
                                    line += "&"
                                    lines.append(line)
                                    line = "            "
                                    line_length = len(line)
                            else:
                                line += parameter_name + ") &"
                                lines.append(line)
                                lines.append("            result(success)")
                        else:
                            sys.exit("Error: <parameter> -tags must have attribute 'name'")

        return lines
    
    @staticmethod
    def get_tag_lines(tag, core_object_name, abbreviation = None, global_index_counter = None, class_name = None, tree_level = 0):
        lines = []
        if global_index_counter is None:
            if 'global_index_counter' in tag.attrib:
                global_index_counter = tag.attrib['global_index_counter']
                
        if tag.tag == 'settings_input':
            class_name = "settings"
            
        local_index_counter = None
        if SettingsGenerator.generate_fortran(tag):
            if 'name' in tag.attrib:
                name = tag.attrib['name']
                if 'abbreviation' in tag.attrib:
                    abbreviation = tag.attrib['abbreviation']
                if 'local_index_counter' in tag.attrib:
                    local_index_counter = tag.attrib['local_index_counter']
                if 'fortran_variable_name' in tag.attrib:
                    name = tag.attrib['fortran_variable_name']
                    
                for parameter in tag.findall('parameter'):
                    if SettingsGenerator.generate_fortran(parameter):
                        lines.extend(SettingsGenerator.get_interface_array_lines(parameter, core_object_name, name, class_name, abbreviation, global_index_counter, local_index_counter, tree_level))
                        
                for subtag in tag.findall('class'):
                    lines.extend(SettingsGenerator.get_tag_lines(subtag, core_object_name, abbreviation = abbreviation,  global_index_counter = global_index_counter, class_name = class_name, tree_level = 0))
            else:
                sys.exit("Error: <{}> -tags must have attribute 'name'. Attributes: {}".format(tag.tag, tag.attrib))
        return lines
    
    @staticmethod
    def get_interface_array_lines(parameter, core_object_name, type_name, class_name, parent_abbreviation, global_index_counter, local_index_counter, tree_level):
        lines = []
        indeces = ['i', 'j', 'k', 'l', 'm']
        
        parameter_name = parameter.attrib['name']
        input_parameter_name = SettingsGenerator.get_parameter_name(parameter.attrib['name'], parent_abbreviation)
        #shape = parameter.attrib['shape']
        counter_name = "{}counter".format(indeces[tree_level])
        if class_name is not None:
            object_variable_name = "{}%{}({})%{}%{}".format(core_object_name, class_name, indeces[tree_level], type_name, parameter_name)
        else:
            object_variable_name = "{}%{}({})%{}".format(core_object_name, type_name, indeces[tree_level], parameter_name)
        
        tree_indent = "        "
        for i in range(tree_level):
            tree_indent += "    "
        indent = tree_indent
        if not SettingsGenerator.is_required(parameter):
            lines.append("{}if (present({})) then".format(tree_indent, input_parameter_name))
            indent += "    "
        if local_index_counter is None:
            increment = "1"
            if SettingsGenerator.is_array(parameter):
                index = "(:, {})".format(counter_name)
            else:
                index = "({})".format(counter_name)
        else:
            local_counter_name = SettingsGenerator.get_parameter_name(local_index_counter, parent_abbreviation)
            if SettingsGenerator.is_array(parameter):
                index = "(:, {} : {} + {}({})-1)".format(counter_name, counter_name, local_counter_name, indeces[tree_level])
            else:
                index = "({} : {} + {}({})-1)".format(counter_name, counter_name, local_counter_name, indeces[tree_level])
            increment = "{}({})".format(local_counter_name, indeces[tree_level])
        #lines.append("        allocate({}%{}(number_of_structures))".format(core_object_name, type_name))
        lines.append("{}{} = 1".format(indent, counter_name))
        lines.append("{}do {} = 1, {}".format(indent, indeces[tree_level], global_index_counter)) 
        lines.append("{}    {} = &".format(indent, object_variable_name))
        lines.append("{}        {}{}".format(indent, input_parameter_name, index)) 
        lines.append("{}    {} = {} + {}".format(indent, counter_name, counter_name, increment)) 
        lines.append("{}end do".format(indent)) 
        
        if not SettingsGenerator.is_required(parameter):
            lines.append("{}end if".format(tree_indent, input_parameter_name))
        
        return lines
    
    @staticmethod
    def get_sub_classes(tag, result_sub_classes = []):
        sub_classes = tag.findall('class')
        for sub_class in sub_classes:
            result_sub_classes.append(sub_class)
            SettingsGenerator.get_sub_classes(sub_class, result_sub_classes)
        return result_sub_classes
    
            
    @staticmethod
    def get_interface_parameter_lines(parameter, parent_abbreviation, shape_parameter_name):
        lines = []
        if SettingsGenerator.generate_fortran(parameter):
            if 'name' in parameter.attrib:
                name = parameter.attrib['name']
                if 'type' in parameter.attrib:
                    data_type = parameter.attrib['type']
                    lines.extend(SettingsGenerator.get_comment_lines(parameter, name))
                    if data_type.startswith('bool'):
                        lines.extend(SettingsGenerator.get_bool_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name))
                    elif data_type.startswith('int'):
                        lines.extend(SettingsGenerator.get_int_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name))
                    elif data_type.startswith('float'):
                        lines.extend(SettingsGenerator.get_float_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name))
                    elif data_type.startswith('double'):
                        lines.extend(SettingsGenerator.get_double_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name))
                    elif data_type.startswith('string'):
                        lines.extend(SettingsGenerator.get_string_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name))
                    else:
                        sys.exit("Error: <parameter> -tag with name '{}' has invalid 'type': '{}'. Allowed types are 'bool', 'int', 'float', 'double', 'string' and corresponding array types 'bool[]', 'int[]', 'float[]', 'double[]' and 'string[]'.".format(name, data_type))
                else:
                    
                    sys.exit("Error: <parameter> -tags must have attribute 'type'. Attributes: {}".format(parameter.attrib))
            else:
                sys.exit("Error: <parameter> -tags must have attribute 'name'")
        return lines
    
    @staticmethod
    def generate_fortran(tag):
        return 'generate_fortran' not in tag.attrib or tag.attrib['generate_fortran'].lower()  == 'true'
    
    @staticmethod
    def is_required(tag):
        if 'required' in tag.attrib:
            if tag.attrib['required'].lower() == "true" or tag.attrib['required'] == 1:
                return True
        return False
    
    @staticmethod
    def is_array(parameter):
        if 'type' in parameter.attrib:
            if parameter.attrib['type'].endswith("[]"):
                return True
        return False
    
    @staticmethod
    def is_valid_parameter(parameter, errors = None):
        result = True
        parameter_name = None
        if 'name' not in parameter.attrib:
            parameter_name = parameter.attrib['name']
            if errors is not None:
                errors.append("<parameter> -tags must have attribute 'name'")
            result = False
            
        if 'type' not in parameter.attrib:
            if errors is not None:
                errors.append("<parameter> -tags must have attribute 'type'")
            result = False
        else:
            result = SettingsGenerator.is_valid_parameter_type(parameter.attrib['type'], errors = errors, parameter_name = parameter_name)
        return result
    
    @staticmethod
    def has_options(parameter):
        first_option = parameter.find('option')
        return first_option != None
       
    
    @staticmethod
    def is_valid_parameter_type(parameter_type, errors = None, parameter_name = None):
        result = True
        if not (parameter_type.startswith('bool') or parameter_type.startswith('double') or parameter_type.startswith('float') or parameter_type.startswith('int') or parameter_type.startswith('string')):
            result = False
            print("parameter type: ", parameter_type)
            if errors is not None:
                errors.append("Error: <parameter> -tag with name '{}' has invalid 'type': '{}'. Allowed types are 'bool', 'int', 'float', 'double', 'string' and corresponding array types 'bool[]', 'int[]', 'float[]', 'double[]' and 'string[]'.".format(parameter_name, parameter_type))
        return result
    
    
    @staticmethod
    def get_interface_basis_set_lines(parameter, core_object_name, parent_name, parent_abbreviation):
        lines = []
        errors = []
        if SettingsGenerator.generate_fortran(parameter):
            if SettingsGenerator.is_valid_parameter(parameter, errors):
                name = parameter.attrib['name']
                lines.append("        if (present({}_{})) then".format(parent_abbreviation, name))
                lines.append("            do i = 1, number_of_basis_sets") 
                lines.append("                {}%basis_sets(i)%{}%{} = &".format(core_object_name, parent_name, name))
                lines.append("                    {}_{}".format(parent_abbreviation, name))
                lines.append("            end do") 
                lines.append("        end if")
            else:
                sys.exit(errors[0])
        return lines

    @staticmethod
    def get_option_lines(parameter, is_in_subclass = False):
        lines = []
        # find all option tags child to this parameter
        found_options = parameter.findall("option")
        if found_options is not None and len(found_options) > 0:
            # go through all options
            for option in found_options:
                # and select the ones that have the attribute 'fortran_option_name'
                if 'fortran_option_name' in option.attrib and 'value' in option.attrib:
                    lines.append("    integer, parameter :: {} = {}".format(option.attrib['fortran_option_name'], option.attrib['value']))
        return lines
            
            
    @staticmethod
    def get_parameter_lines(parameter, is_in_subclass = False):
        lines = []
        if SettingsGenerator.generate_fortran(parameter):
            if 'name' in parameter.attrib:
                name = parameter.attrib['name']
                if 'type' in parameter.attrib:
                    data_type = parameter.attrib['type']
                    lines.extend(SettingsGenerator.get_comment_lines(parameter, name))
                    if data_type.startswith('bool'):
                        lines.append(SettingsGenerator.get_bool_line(parameter, name, is_in_subclass))
                    elif data_type.startswith('int'):
                        lines.append(SettingsGenerator.get_int_line(parameter, name, is_in_subclass))
                    elif data_type.startswith('float'):
                        lines.append(SettingsGenerator.get_double_line(parameter, name, is_in_subclass))
                    elif data_type.startswith('double'):
                        lines.append(SettingsGenerator.get_double_line(parameter, name, is_in_subclass))
                    elif data_type.startswith('string'):
                        lines.append(SettingsGenerator.get_string_line(parameter, name, is_in_subclass))
                    else:
                        print("parameter type", data_type)
                        sys.exit("Error: <parameter> -tag with name '{}' has invalid 'type': '{}'. Allowed types are 'bool', 'int', 'float', 'double', 'string' and corresponding array types 'bool[]', 'int[]', 'float[]', 'double[]' and 'string[]'.".format(name, data_type))
                else:
                    sys.exit("Error: <parameter> -tags must have attribute 'type'")
            else:
                sys.exit("Error: <parameter> -tags must have attribute 'name'")
        return lines
   
    
    @staticmethod
    def get_comment_lines(tag, name, indent="        "):
        lines = []
        if 'comment' in tag.attrib:
            comment_words = tag.attrib['comment'].split(" ")
            total_length = 0
            line = ""
            first_line = True
            for word in comment_words:
                total_length += len(word)+1
                line += "{} ".format(word)
                if total_length > 70:
                    if first_line:
                        lines.append("{}!> {}".format(indent, line))
                        first_line = False
                    else:
                        lines.append("{}!! {}".format(indent, line))
                    total_length = 0
                    line = ""
            
            if first_line:
                lines.append("{}!> {}".format(indent, line))
            else:
                lines.append("{}!! {}".format(indent, line))
        else:
            print("Warning: tag of type '{}' with name '{}' does not have a comment. Having a comment is highly recommended.".format(tag.tag, name))
        return lines
            
   
    @staticmethod
    def get_bool_line(parameter, name, is_in_subclass = False):
        if is_in_subclass:
            if SettingsGenerator.is_array(parameter):
                return "        logical,      allocatable :: {}(:, :)".format(name)
            else:
                return "        logical,      allocatable :: {}(:)".format(name)
        else:
            if SettingsGenerator.is_array(parameter):
                return "        logical,      allocatable :: {}(:)".format(name)
            
            default_value = '.FALSE.'
            if 'default' in parameter.attrib:
                if parameter.attrib['default'].lower() == 'true' or parameter.attrib['default'] == '1':
                    default_value = '.TRUE.' 
                else:
                    default_value = '.FALSE.'
            else:
                default_value = '.FALSE.'
                print("Warning: parameter with name '{}' does not have default value.".format(name))
                
            return "        logical              :: {} = {}".format(name, default_value)
    
    @staticmethod
    def get_int_line(parameter, name, is_in_subclass = False):
        if is_in_subclass:
            if SettingsGenerator.is_array(parameter):
                return "        integer,      allocatable :: {}(:, :)".format(name)
            else:
                return "        integer,      allocatable :: {}(:)".format(name)
        else:
            if SettingsGenerator.is_array(parameter):
                return "        integer,      allocatable :: {}(:)".format(name)
            
            default_value = 0
            if 'default' in parameter.attrib:
                try:
                    default_value = int(parameter.attrib['default'])
                except ValueError:
                    sys.exit('Error: <parameter> with type int and name "{}" has invalid default value: {}'.format(name, parameter.attrib['default']))
            else:
                default_value = '0'
                print("Warning: parameter with name '{}' does not have default value.".format(name))
                
            
            return "        integer                   :: {} = {}".format(name, default_value)
    
    @staticmethod
    def get_float_line(parameter, name, is_in_subclass = False):
        if is_in_subclass:
            if SettingsGenerator.is_array(parameter):
                return "        real(REAL64), allocatable :: {}(:, :)".format(name)
            else:
                return "        real(REAL64), allocatable :: {}(:)".format(name)
        else:
            if SettingsGenerator.is_array(parameter):
                return "        real(REAL64), allocatable :: {}(:)".format(name)
            
            default_value = 0
            if 'default' in parameter.attrib:
                try:
                    default_value = float(parameter.attrib['default'])
                except ValueError:
                    sys.exit('Error: <parameter> with type float and name "{}" has invalid default value: {}'.format(name, parameter.attrib['default']))
            else:
                default_value = '0.0'
                print("Warning: parameter with name '{}' does not have default value.".format(name))
                
            return "        real(REAL64)              :: {0} = {1:.10f}".format(name, default_value)
    
    @staticmethod
    def get_double_line(parameter, name, is_in_subclass = False):
        if is_in_subclass:
            if SettingsGenerator.is_array(parameter):
                return "        real(REAL64), allocatable :: {}(:, :)".format(name)
            else:
                return "        real(REAL64), allocatable :: {}(:)".format(name)
        else:
            if SettingsGenerator.is_array(parameter):
                return "        real(REAL64), allocatable :: {}(:)".format(name)
            
            default_value = 0
            if 'default' in parameter.attrib:
                try:
                    default_value = float(parameter.attrib['default'])
                except ValueError:
                    sys.exit('Error: <parameter> with type double and name "{}" has invalid default value: {}'.format(name, parameter.attrib['default']))
            else:
                default_value = '0.0d0'
                print("Warning: parameter with name '{}' does not have default value.".format(name))
            
            return "        real(REAL64)              :: {0} = {1:.10f}d0".format(name, default_value)
    
    @staticmethod
    def get_string_line(parameter, name, is_in_subclass = False):
        if is_in_subclass:
            if SettingsGenerator.is_array(parameter):
                return "        character*256, allocatable :: {}(:, :)".format(name)
            else:
                return "        character*256, allocatable :: {}(:)".format(name)
        else:
            if SettingsGenerator.is_array(parameter):
                return "        character*256, allocatable :: {}(:)".format(name)
            
            default_value = 0
            if 'default' in parameter.attrib:
                try:
                    default_value = str(parameter.attrib['default'])
                except ValueError:
                    sys.exit('Error: <parameter> with type string and name "{}" has invalid default value: {}'.format(name, parameter.attrib['default']))
            else:
                default_value = ''
                print("Warning: parameter with name '{}' does not have default value.".format(name))
                
            return "        character*256              :: {} = \"{}\"".format(name, default_value)
    
    @staticmethod
    def get_bool_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name):
        type_string = "{}".format("logical,        ")
        return SettingsGenerator.get_parameter_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name, type_string)
    
    @staticmethod
    def get_int_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name):
        
        type_string = "{}".format("integer,        ")
        return SettingsGenerator.get_parameter_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name, type_string)
    
    @staticmethod
    def get_float_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name):
        type_string = "{}".format("real(8),        ")
        return SettingsGenerator.get_parameter_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name, type_string)
    
    @staticmethod
    def get_double_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name):
        type_string = "{}".format("real(8),        ")
        return SettingsGenerator.get_parameter_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name, type_string)
    
    
    @staticmethod
    def get_string_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name):
        type_string = "{}".format("character*256, ")
        return SettingsGenerator.get_parameter_interface_lines(parameter, name, data_type, parent_abbreviation, shape_parameter_name, type_string)
    
    @staticmethod
    def handle_shape(parameter, data_type, parent_abbreviation, global_index_counter):
        if data_type.endswith("[]"):
            if 'shape' in parameter.attrib:
                shape_attrib = parameter.attrib['shape'] 
                shape_list = shape_attrib.strip().split(",")
                depend = None
                shape = None
                for arg in shape_list:
                        
                    # hande the effects of shape on depend -list
                    # pure ints do not need a dependency
                    try:
                        int(arg.strip())
                        # handle the shape int parameter
                        if shape is None:
                            shape = arg.strip()
                        else:
                            shape +=  ", {}".format(arg.strip())
                
                    except ValueError:
                        parameter_name = SettingsGenerator.get_parameter_name(arg.strip(), parent_abbreviation)
                        # handle the shape string parameter
                        if shape is None:
                            shape = "{}+1".format(parameter_name)
                        else:
                            shape +=  ", {}+1".format(parameter_name)
                        if depend is None:
                            depend = parameter_name
                        else:
                            depend += ", {}".format(parameter_name)
                   
                # add the global index counter to the shape  and depend
                if global_index_counter is not None:
                    if shape is None:
                        shape = "{}+1".format(global_index_counter)
                    else:
                        shape += ", {}+1".format(global_index_counter)
                        
                    if depend is None:
                        depend = global_index_counter
                    else:
                        depend += ", {}".format(global_index_counter)
        else:
            # if shape is not there, the only thing that matters is the global index counter
            if global_index_counter is not None:
                shape = "{}+1".format(global_index_counter)
                depend = global_index_counter
            else:
                shape = None
                depend = None
        return depend, shape
    
    @staticmethod
    def get_parameter_name(name, parent_abbreviation):
        if parent_abbreviation is not None:
            return "{}_{}".format(parent_abbreviation, name)
        else:
            return name
    
    @staticmethod
    def get_parameter_interface_lines(parameter, name, data_type, parent_abbreviation, global_index_counter, type_string):
        lines = []
        depend, shape = SettingsGenerator.handle_shape(parameter, data_type, parent_abbreviation, global_index_counter)
        # initialize the line
        f2py_line = "        !f2py {}intent(in)".format(type_string)
        fortran_line = "        {}intent(in)".format(type_string)
        
        # set the dimension
        if shape is not None:
            f2py_line += ", dimension({})".format(shape)
        else:
            f2py_line += "               "
            
        # add possibly the dependencies to the f2py line
        if depend is not None:
            f2py_line += ", depend({})".format(depend)
        else:
            f2py_line += "            "
        
        # add possibly the optional flag
        if SettingsGenerator.is_required(parameter):
            f2py_line += "          "
            fortran_line += "          "
        else:
            f2py_line += ", optional"
            fortran_line += ", optional"
            
        # finally add the parameter name (with or without the parent abbreviation)
        parameter_name = SettingsGenerator.get_parameter_name(name, parent_abbreviation)
        f2py_line += " :: {}".format(parameter_name)
        fortran_line += " :: {}".format(parameter_name)
        
        if shape is not None:
            fortran_line += "({})".format(shape)
        
        # we are finished, add the lines to the result array
        lines.append(f2py_line)
        lines.append(fortran_line)
        return lines
    
    @staticmethod
    def get_auto_generated_lines():
        lines = []
        lines.append("!    NOTE: This file is automatically generated by the python program              !")
        lines.append("!    'generate_objects.py'. DO NOT MODIFY THIS FILE manually. To change            !")
        lines.append("!    the contents of this file, edit the input_parameters.xml file and run         !")
        lines.append("!    'python generate_objects.py' in the 'python/' folder.                         !")
        return lines
    
    @staticmethod
    def get_license_lines():
        lines = []
        lines.append("!----------------------------------------------------------------------------------!")
        lines.append("!    Copyright (c) 2010-2018 Pauli Parkkinen, Eelis Solala, Wen-Hua Xu,            !")                      
        lines.append("!                            Sergio Losilla, Elias Toivanen, Jonas Juselius        !")
        lines.append("!                                                                                  !")
        lines.append("!    Permission is hereby granted, free of charge, to any person obtaining a copy  !")
        lines.append('!    of this software and associated documentation files (the "Software"), to deal !')
        lines.append("!    in the Software without restriction, including without limitation the rights  !")
        lines.append("!    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     !")
        lines.append("!    copies of the Software, and to permit persons to whom the Software is         !")
        lines.append("!    furnished to do so, subject to the following conditions:                      !")
        lines.append("!                                                                                  !")
        lines.append("!    The above copyright notice and this permission notice shall be included in all!")
        lines.append("!    copies or substantial portions of the Software.                               !")
        lines.append("!                                                                                  !")
        lines.append('!    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    !')
        lines.append("!    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      !")
        lines.append("!    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   !")
        lines.append("!    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        !")
        lines.append("!    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, !")
        lines.append("!    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE !")
        lines.append("!    SOFTWARE.                                                                     !")
        lines.append("!----------------------------------------------------------------------------------!")
        return lines
    
    @staticmethod
    def get_divider_lines():
        lines = []
        lines.append("!----------------------------------------------------------------------------------!")
        return lines
    
def is_valid_input_definition(self, input_definition_path):
    gen = SettingsGenerator(input_definition_path)
    

def read_in_template(template_filename):
    template = None
    if os.path.exists(template_filename):
        template = {"procedures": [], "constructors": [], "content": []}
        with open(template_filename) as f:
            all_lines = f.readlines()
            
        try:
            for line in all_lines:
                if line.startswith("---------------------PROCEDURES"):
                    target = template["procedures"]
                elif line.startswith("---------------------CONSTRUCTORS"):
                    target = template["constructors"]
                elif line.startswith("---------------------CONTENT"):
                    target = template["content"]
                else:
                    target.append(line.replace(os.linesep, ""))
        except:
            raise InvalidTemplateException(\
                "Invalid template: Template file should"+\
                " have PROCEDURES, CONSTRUCTORS and CONTENT sections.")
    return template
                
        
    
        

def write_file(filename, lines):
    f = open(filename,'w')
    for line in lines:
        f.write(line+"\n")
    f.close()

if __name__ == "__main__":
    gen = SettingsGenerator(os.path.dirname(os.path.realpath(__file__))+"/input_parameters.xml")
    write_file(os.path.dirname(os.path.realpath(__file__))+"/../src/bubbles/settings.F90",lines = gen.generate_settings())
    write_file(os.path.dirname(os.path.realpath(__file__))+"/../src/bubbles/action.F90",lines = gen.generate_action())
    write_file(os.path.dirname(os.path.realpath(__file__))+"/../src/bubbles/scf_energetics.F90",\
               lines = gen.generate_scf_energetics(\
                   template = read_in_template(os.path.dirname(os.path.realpath(__file__))+"/../src/bubbles/scf_energetics.F90.template")) \
               )
    write_file(os.path.dirname(os.path.realpath(__file__))+"/../utils/python_interface.F90", gen.generate_interface())
                
