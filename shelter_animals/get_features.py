#!/usr/bin/env python
import os, sys
import numpy as np
import pandas as pd


OUTPUT_HEADER = 'ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer'
OUTPUT_ROW = [
    ',0,1,0,0,0',   # Died
    ',0,0,1,0,0',   # Euthanasia
    ',0,0,0,0,1',   # Transfer
    ',1,0,0,0,0',   # Adoption
    ',0,0,0,1,0',   # Return to owner
]


def enum_outcome(outcome):
    if outcome == 'Return_to_owner':
        return 4
    if outcome == 'Adoption':
        return 3
    if outcome == 'Transfer':
        return 2
    if outcome == 'Euthanasia':
        return 1
    if outcome == 'Died':
        return 0
    raise Exception


def convert_age_to_days(age_string):
    if type(age_string) != type(''):
        return age_string
    num, unit = age_string.split()
    if unit[0] == 'd':   # I don't expect it, but just in case
        return int(num)
    if unit[0] == 'w':
        return int(num) * 7
    if unit[0] == 'm':
        return int(num) * 30
    if unit[0] == 'y':
        return int(num) * 365


def add_more_columns(data):
    data['Cat'] = data['AnimalType'].apply(lambda x: x.lower() == 'cat')
    data['Sex'] = data['SexuponOutcome'].apply(lambda x: x.split()[1] if (x != 'Unknown' and type(x) == type('')) else None)
    data['Sex'] = data['Sex'].apply(lambda x: True if x == 'Female' else (False if x == 'Male' else None))
    data['Sterilisation'] = data['SexuponOutcome'].apply(lambda x: not x.split()[0]=='Intact' if (x != 'Unknown' and type(x) == type('')) else None)
    data['AgeInDays'] = data['AgeuponOutcome'].apply(convert_age_to_days)
    # names
    data['HasName'] = data['Name'].apply(lambda x: not pd.isnull(x))
    data['NameMax'] = data['Name'].apply(lambda x: 'max' in x.lower() if not pd.isnull(x) else False)
    data['NameBella'] = data['Name'].apply(lambda x: 'bella' in x.lower() if not pd.isnull(x) else False)
    # breeds
    data['Mix'] = data['Breed'].apply(lambda x: x.endswith('Mix'))
    data['Domestic'] = data['Breed'].apply(lambda x: 'domestic' in x.lower())
    data['Shorthair'] = data['Breed'].apply(lambda x: 'shorthair' in x.lower())
    data['Longhair'] = data['Breed'].apply(lambda x: 'longhair' in x.lower())
    data['Siamese'] = data['Breed'].apply(lambda x: 'siamese' in x.lower())
    data['PitBull'] = data['Breed'].apply(lambda x: 'pit bull' in x.lower())
    data['Australian'] = data['Breed'].apply(lambda x: 'australian' in x.lower())
    data['Retriever'] = data['Breed'].apply(lambda x: 'retriever' in x.lower())
    data['Shepherd'] = data['Breed'].apply(lambda x: 'shepherd' in x.lower())
    data['Terrier'] = data['Breed'].apply(lambda x: 'terrier' in x.lower())
    data['Chihuahua'] = data['Breed'].apply(lambda x: 'chihuahua' in x.lower())
    # colors
    data['Black1'] = data['Color'].apply(lambda x: 'black' in x.lower())
    data['Black2'] = data['Color'].apply(lambda x: x.lower() == 'black')
    data['White'] = data['Color'].apply(lambda x: 'white' in x.lower())
    data['Tabby'] = data['Color'].apply(lambda x: 'tabby' in x.lower())
    data['Tiger'] = data['Color'].apply(lambda x: 'tiger' in x.lower())
    data['Blue'] = data['Color'].apply(lambda x: 'blue' in x.lower())
    data['Brown'] = data['Color'].apply(lambda x: 'brown' in x.lower())
    data['Orange'] = data['Color'].apply(lambda x: 'orange' in x.lower())
    data['Red'] = data['Color'].apply(lambda x: 'red' in x.lower())
    data['Yellow'] = data['Color'].apply(lambda x: 'yellow' in x.lower())
    data['Tan'] = data['Color'].apply(lambda x: 'tan' in x.lower())
    data['Tricolor'] = data['Color'].apply(lambda x: 'tricolor' in x.lower())
    data['2colors'] = data['Color'].apply(lambda x: '/' in x)


def select_columns(data):
    result = data.copy()
    result = pd.DataFrame(result,
                              columns=('Cat', 'Sex', 'Sterilisation', 'AgeInDays',
                                       'HasName', 'NameMax', 'NameBella',
                                       'Mix', 'Domestic', 'Shorthair', 'Longhair', 'Siamese',
                                       'PitBull', 'Australian', 'Retriever', 'Shepherd', 'Terrier', 'Chihuahua',
                                       'Black1', 'Black2', 'White', 'Tabby', 'Tiger', 'Blue', 'Brown',
                                       'Orange', 'Red', 'Yellow', 'Tan', 'Tricolor', '2colors'
                                      )
                             )
    return result.values


def get_num_features(data):
    add_more_columns(data)
    return select_columns(data)

