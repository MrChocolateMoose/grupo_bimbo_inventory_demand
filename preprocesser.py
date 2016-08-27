import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def preprocess_producto_tabla(producto_df):

    # https://regex101.com/

    unprocessed_product_name = 'NombreProducto'

    #capture characters till numeric is found. Don't include last space
    producto_df['product_name'] = producto_df[unprocessed_product_name].str.extract('^(\D*) ', expand=False)

    # skip over product name and weight and capture the brand acronym. Don't include the product number:  '^\D*\d*\D* (\D*) \d'
    producto_df['brand'] = producto_df[unprocessed_product_name].str.extract('^.+\s(\D+) \d+$', expand=False)

    weight_pair = producto_df[unprocessed_product_name].str.extract('(\d+)(Kg|g)', expand=True)
    #assert(len(weight_pair) == 2)

    producto_df['weight'] = weight_pair[0].astype('float') * weight_pair[1].map({'Kg' : 1000, 'g' : 1})

    producto_df['pieces'] =  producto_df[unprocessed_product_name].str.extract('(\d+)p ', expand=False).astype('float')


    # reduction from nunique of 1010 to nunique of 990
    producto_df['product_name'] = producto_df['product_name'].apply(
            lambda x: " ".join([i for i in str(x).lower().split() if i not in stopwords.words("spanish")])
    )

    stemmer = SnowballStemmer("spanish")

    # reduction from nunique of 990 to nunique of 961
    producto_df['product_name'] = producto_df['product_name'].apply(
        lambda x: " ".join([stemmer.stem(i) for i in str(x).lower().split()])
    )

    #producto_df.drop(unprocessed_product_name, axis=1, inplace=True)


    return producto_df