import pymysql
import os
import pandas as pd
import logging


def get_connection():
    """Returns a connection with a database using the environment variables

    Returns
    -------
        A PyMYSQL connection
    """

    return pymysql.connect(host=os.environ.get('DB_HOST'),
                           port=int(os.environ.get('DB_PORT')),
                           user=os.environ.get('DB_USER'),
                           password=os.environ.get('DB_PASS'),
                           db=os.environ.get('DB_NAME'),
                           cursorclass=pymysql.cursors.DictCursor)


def get_data_sql(query):
    """Acquire a dataset from a sql query either from prod or dev database.

    For security purposes, some of the parameters for the connection are
    environment variables and should be set when configuring the conda env.
    For more information, check the README.
    """

    connection = get_connection()
    data = pd.read_sql(query, connection)

    return data
