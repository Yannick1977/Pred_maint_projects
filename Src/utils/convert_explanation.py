import re
import numpy as np

class explanation_convertion:

    def __init__(self) -> None:
        pass

    def trouver_numeriques(self,chaine):
        """
        Finds all the numeric values in a given string.

        Parameters:
        chaine (str): The input string to search for numeric values.

        Returns:
        list: A list of all the numeric values found in the input string.
        """
        return re.findall(r'\d+', chaine)

    def trouver_flottants(self,chaine):
        """
        Finds and returns all floating-point numbers in a given string.

        Parameters:
        chaine (str): The input string to search for floating-point numbers.

        Returns:
        list: A list of floating-point numbers found in the input string.
        """
        return re.findall(r'\d+\.\d+', chaine)

    def split_string(self, string, delimiters):
        """
        Splits a string based on the given delimiters.

        Args:
            string (str): The input string to be split.
            delimiters (list): A list of delimiters to split the string.

        Returns:
            list: A list of substrings obtained after splitting the string.

        Example:
            >>> split_string("Hello, World! How are you?", [",", " "])
            ['Hello', 'World!', 'How', 'are', 'you?']
        """
        delimiters = "|".join(map(re.escape, delimiters))
        return [s for s in re.split(delimiters, string) if s]

    def inverse_transform(self, ct, scaler_name, col_name, value):
        """
        Inverse transforms a given value for a specific column using the specified scaler.

        Parameters:
        ct (ColumnTransformer): The ColumnTransformer object used for feature transformation.
        scaler_name (str): The name of the scaler within the ColumnTransformer.
        col_name (str): The name of the column to inverse transform.
        value: The value to inverse transform.

        Returns:
        float: The inverse transformed value for the specified column.
        """
        cols = np.array(ct.named_transformers_[scaler_name].get_feature_names_out())
        idx_cols = np.where(cols == col_name)[0][0]
        n = len(ct.named_transformers_[scaler_name].get_feature_names_out())
        empty_np = np.empty((n, 1))
        empty_np[idx_cols] = value
        inv_ct = ct.named_transformers_[scaler_name].inverse_transform(empty_np.T)
        return inv_ct[:, idx_cols][0]


    def convert_data_explanation(self, lst_exp, ct):
        """
        Converts the explanations in the given list to a more readable format.

        Args:
            lst_exp (list): List of explanations to be converted.
            ct: The scaler used for inverse transformation.

        Returns:
            list: List of converted explanations, where each explanation is a tuple
                containing the modified string and the corresponding weight.
        """
        delimiteurs = ['>=', '<=', '>', '<', '!=', '==']
        ret = []
        for expl in lst_exp:
            lst_expl_details = self.split_string(expl[0], delimiteurs)
            if len(lst_expl_details) == 2:
                tmp = self.split_string(lst_expl_details[0], ['__'])
                #print("tmp: "+str(tmp))
                if (tmp[0].strip() == 'cat'):
                    a = self.trouver_flottants(lst_expl_details[1])[0]
                else:
                    a = round(self.inverse_transform(ct, tmp[0].strip(), tmp[1].strip(), lst_expl_details[1]), 2)
                st1 = (expl[0]).replace(lst_expl_details[1], str(a))    # remplacement de la valeur par la valeur réelle
                st1  = st1.replace('cat__', '').replace('num__', '')    # suppression des préfixes   
                z = (st1, expl[1]*100)

            if len(lst_expl_details) == 3:
                tmp = self.split_string(lst_expl_details[1], ['__'])
                #print(tmp)
                if (tmp[0].strip() == 'cat'):
                    a = self.trouver_flottants(lst_expl_details[0])[0]
                    b = self.trouver_flottants(lst_expl_details[2])[0]
                else:
                    a = round(self.inverse_transform(ct, tmp[0].strip(), tmp[1].strip(), lst_expl_details[0]), 2)
                    b = round(self.inverse_transform(ct, tmp[0].strip(), tmp[1].strip(), lst_expl_details[2]), 2)
                st1 = (expl[0]).replace(lst_expl_details[0], str(a)).replace(lst_expl_details[2], str(b))   #remplacement de la valeur par la valeur réelle
                st1 = st1.replace('cat__', '').replace('num__', '') # suppression des préfixes
                z = (st1, expl[1]*100)
            ret.append(z)
        return ret
