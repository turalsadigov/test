import pandas as pd
import polars as pl
import sshtunnel
import pymysql
import time
from dotenv import load_dotenv
import dotenv
import os
import requests
import json
import numpy as np
import pprint
import streamlit as st

from requests.auth import HTTPBasicAuth


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


load_dotenv()


# Load the environment variables
# define enpoints
dotenv.load_dotenv()
AWS_MAGICPORT_SERVER_URL = os.environ.get("AWS_MAGICPORT_SERVER_URL")
EC2_STATIC_IP = os.environ.get("EC2_STATIC_IP")
EC2_USER = os.environ.get("EC2_USER")

vessels_merged_list_url = "http://" + AWS_MAGICPORT_SERVER_URL + "/Vessels/MergedList"
vessels_journey_url = "http://" + AWS_MAGICPORT_SERVER_URL + "/Vessels/Journey"
current_directory = os.getcwd()

# DB related
def get_human_readable_time(seconds):
    if seconds < 1:
        milliseconds = seconds * 1000
        return f"{milliseconds:.2f} milliseconds"
    elif seconds > 60:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        return f"{seconds:.3f} seconds"


def get_magicport_response(
    database_location="liquidweb",
    database_name="magicportai_core",
    query="select * from agn_berth_types",
    prod = True
):
    if prod:
        user = "magicportai_tural"
        database_ip = "67.227.130.50"
        # if database name does not comntain letters ai, return message
        if "ai" not in database_name:
            return pd.DataFrame({"Error:": ["Database name does not contain 'ai'"]})
    else:
        user = "magicport_tural"
        database_ip = "67.227.130.237"
        if "ai" in database_name:
            return pd.DataFrame({"Error:": ["Remove 'ai' from database name"]})
    start_time = time.time()
    if database_location == "liquidweb":
        db_params = {
            "user": user,
            "password": "Hk.=}=Iqq1he",
            "host": "localhost",
            "port": 4000,
            "database": database_name,
        }
        with sshtunnel.open_tunnel(
            (EC2_STATIC_IP, 22),
            ssh_username=EC2_USER,
            ssh_pkey="magicport_database_access.pem",
            remote_bind_address=(
                database_ip,
                3306,
            ),
            local_bind_address=("0.0.0.0", 4000),
        ) as tunnel:
            try:
                print(f"SSH tunnel established.")
                conn = pymysql.connect(**db_params)
                print("MySQL connection established")
                cur = conn.cursor()
                cur.execute(query)  # put whatever query you want here!
                result = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]
                print("Query executed successfully.")
                cur.close()
                conn.close()
                data_list = []
                for row in result:
                    data_dict = dict(zip(column_names, row))
                    data_list.append(data_dict)
                try:
                    print("Creating Pandas DataFrame instead.")
                    df = pd.DataFrame(data_list)
                    #print("Polars Dataframe created successfully.")
                except pl.exceptions.ComputeError:
                    #print("Polars Error: ", pl.exceptions.ComputeError)
                    print("Creating Pandas DataFrame instead.")
                    df = pd.DataFrame(data_list)
            except pymysql.MySQLError as err:
                print(f"MySQL Error: {err}")
                df = pl.DataFrame({"MySQL Error:": [err]})
    elif database_location == "aws":
        print("AWS connection not implemented yet.")
        #print("Returning a placeholder Polars DataFrame.")
        df = pd.DataFrame({"AWS Message:": ["Not implemented yet."]})
    else:
        print("Invalid database location.")
        #print("Returning a placeholder Polars DataFrame.")
        df = pd.DataFrame({"Invalid Database Location:": [database_location]})
    print(f"Execution time: {get_human_readable_time(time.time() - start_time)}")
    return df


def angle_between_vectors(u, v):
    # Convert lists to numpy arrays
    u = np.array(u)
    v = np.array(v)
    
    # Compute the dot product
    dot_product = np.dot(u, v)
    
    # Compute the magnitudes of the vectors
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_u * norm_v)
    
    # Clamp the cosine value to the range [-1, 1] to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Compute the angle in radians
    angle_rad = np.arccos(cos_theta)
    
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def are_vectors_close(u, v, threshold=0.15):
    # rel error in x xoord
    if max(abs(u[0]), abs(v[0])) != 0:
        x_rel_error = abs(u[0] - v[0]) / max(abs(u[0]), abs(v[0]))
    else:
        x_rel_error = 0
    print("x-rel ERROR: ", x_rel_error)
    if x_rel_error < 0.05:
        return True

    if max(abs(u[1]), abs(v[1])) != 0:
        y_rel_error = abs(u[1] - v[1]) / max(abs(u[1]), abs(v[1]))
    else:
        y_rel_error = 0
    print("y-rel ERROR: ", y_rel_error)
    combined_error = np.sqrt(x_rel_error ** 2 + y_rel_error ** 2)
    print("comb ERROR: ", combined_error)
    if combined_error > threshold:
        return False

    return True


def get_journey_data(imo_mmsi, start, finish, imo=True):
    if imo:
        filter_condition = "imos"
    else:
        filter_condition = "ids"

    headers = {
        "Content-Type": "application/json",
    }

    # Define the parameters
    journey_data = {
        "columns": [
            "id",
            "name",
            "spireIdentity",
            "imo",
            "callSign",
            "shipTypeId",
            "class",
            "flag",
            "a",
            "b",
            "c",
            "d",
            "length",
            "width",
            "maxDraught",
            "dwt",
            "grossTonnage",
            "netTonnage",
            "builtYear",
            "deadYear",
            "hullNumber",
            "shipBuilder",
            "commercialOwner",
            "vesselType",
            "subtype",
            "insertDate",
            "updateDate",
            "unlocode",
            "matchScore",
            "firstLat",
            "lastLat",
            "firstLng",
            "lastLng",
            "firstDraught",
            "lastDraught",
            "firstEta",
            "lastEta",
            "firstDestination",
            "lastDestination",
            "firstSysDate",
            "lastSysDate",
            "firstPositionUpdatedAt",
            "lastPositionUpdatedAt",
            "count",
        ],
        filter_condition: imo_mmsi,
        "start": start,
        "finish": finish,
    }

    # Send the request
    response = requests.post(
        vessels_journey_url, headers=headers, data=json.dumps(journey_data)
    )

    # Check the response
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None

    # cerate a dataframe from the response
    journey_df = pd.DataFrame(response.json())
    # order the rows according to  order by lastPositionUpdatedAt desc
    if not journey_df.empty:
        journey_df = (
            journey_df
            .query("lastSysDate.notnull()")
            .sort_values(by="lastSysDate", ascending=False)
        )
    # query if time is not null

    return journey_df

def get_merged_list_data(imo_mmsi, imo=True):
    if imo:
        data = {"s_staticData_imos": imo_mmsi}
    else:
        data = {"s_staticData_mmsis": imo_mmsi}
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(
        vessels_merged_list_url, headers=headers, data=json.dumps(data)
    )
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None
    merged_df = pd.DataFrame(response.json())
    # order the rows according to  order by s_lastPositionUpdate_timestamp desc
    merged_df = (
        merged_df
        .query("s_lastPositionUpdate_timestamp.notnull()")
        .sort_values(by="s_lastPositionUpdate_timestamp", ascending=False
    ))
    # drop some columns
    merged_df = merged_df.drop(
        columns=[
            "s_id",
            "s_staticData_dimensions_a",
            "s_staticData_dimensions_b",
            "s_staticData_dimensions_c",
            "s_staticData_dimensions_d",
            "s_lastPositionUpdate_accuracy",
            "s_lastPositionUpdate_collectionType",
            "s_lastPositionUpdate_course",
            "s_lastPositionUpdate_heading",
            "s_lastPositionUpdate_rot",
            "s_currentVoyage_draught",
            "s_currentVoyage_matchedPort_port_centerPoint_latitude",
            "s_currentVoyage_matchedPort_port_centerPoint_longitude",
            "s_FromCountryId",
            "s_ToCountryId",
        ]
    )
    # drop all columns starts with s_characteristics_
    # merged_df = merged_df[
    #     [col for col in merged_df.columns if not col.startswith("s_characteristics")]
    # ]
    return merged_df

def get_latest_mmsi(imo):
    data = {
        "columns": [
            "s_staticData_mmsi",
            "s_lastPositionUpdate_timestamp",
            "s_staticData_dimensions_length",
        ],
        "s_staticData_imos": [imo],
    }
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(
        vessels_merged_list_url, headers=headers, data=json.dumps(data)
    )
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None
    pd.DataFrame(response.json()).rename(columns={"s_staticData_mmsi": "mmsi"}).query("s_lastPositionUpdate_timestamp.notnull()").query("s_staticData_dimensions_length >= 70")

    mmsi = (
        pd.DataFrame(response.json())
        .rename(columns={"s_staticData_mmsi": "mmsi"})
        .query("s_lastPositionUpdate_timestamp.notnull()")
        .query("s_staticData_dimensions_length >= 70")
    )
    if mmsi.empty:
        return None
    mmsi = mmsi.sort_values(by="s_lastPositionUpdate_timestamp", ascending=False).head(1)["mmsi"].values[0]
    return mmsi

def get_names_mmsis(name):
    # fix filter below
    data = {
        "query": {
            "query_string": {
                "default_field": "name",
                "query": f"{name}*",
                "default_operator": "AND",
                "allow_leading_wildcard": True
            }
        }
    }
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(
        url="http://67.227.130.50:9200/vessels/_search", 
        auth=HTTPBasicAuth("elastic", "xWWzBE3MPxktg8knkep7"), 
        headers=headers,
        data=json.dumps(data)
    )
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None
    response_json = response.json()
    response_json_hits = response_json["hits"]["hits"]
    pprint.pprint(response_json_hits)
    df = pd.DataFrame(response_json_hits)
    df_source = pd.json_normalize(df['_source'])
    expanded_df = df.drop('_source', axis=1).join(df_source).sort_values(by="s_lastPositionUpdate_timestamp", ascending=False)
    return  expanded_df


def get_merged_list_sister_mmsis(mmsi, imo_given = False, imo = None):
    headers = {
            "Content-Type": "application/json",
        }
    if imo_given:
        corresponding_imo_code = imo
    else:
        data = {"columns": ["s_staticData_imo"], "s_staticData_mmsis": [mmsi]}

        response = requests.post(
            vessels_merged_list_url, headers=headers, data=json.dumps(data)
        )
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            return None
        print(response.json())
        corresponding_imo_code = response.json()[0]["s_staticData_imo"]
    data = {
        "columns": [
            "s_staticData_mmsi",
            "s_staticData_imo",
            "s_staticData_name",
            "s_lastPositionUpdate_timestamp",
            "s_staticData_dimensions_length",
            "s_staticData_dimensions_width",
            "s_staticData_shipType",
        ],
        "s_staticData_imos": [corresponding_imo_code],
    }
    response = requests.post(
        vessels_merged_list_url, headers=headers, data=json.dumps(data)
    )
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None
    sister_mmsis = (
        pd.DataFrame(response.json())
        .rename(columns={"s_staticData_mmsi": "mmsi"})
        .query("s_lastPositionUpdate_timestamp.notnull()")
        .sort_values(by="s_lastPositionUpdate_timestamp", ascending=False)
    )
    return (corresponding_imo_code, sister_mmsis)

def compare_two_length_width(sister_mmsis):
    if sister_mmsis.empty:
        return sister_mmsis
    if sister_mmsis.shape[0] != 2:
        return sister_mmsis
    sister_mmsis = sister_mmsis.sort_values(
            by=["s_lastPositionUpdate_timestamp"],
            ascending=[False],
        )
    small_sister_mmsis = sister_mmsis[['s_staticData_dimensions_length', 's_staticData_dimensions_width']]
    if not small_sister_mmsis.isnull().values.any():
        u = small_sister_mmsis.loc[0, ['s_staticData_dimensions_length', 's_staticData_dimensions_width']].to_numpy()
        v = small_sister_mmsis.loc[1, ['s_staticData_dimensions_length', 's_staticData_dimensions_width']].to_numpy()
        if are_vectors_close(u, v):
            sister_mmsis["mergedlist_cluster"] = 0
        else:
            sister_mmsis.loc[0, "mergedlist_cluster"] = 0
            sister_mmsis.loc[1, "mergedlist_cluster"] = 1
        return sister_mmsis
    return sister_mmsis




# compare_two_length_width(sister_mmsis.head(2).reset_index(drop=True))
# compare_two_length_width(sister_mmsis.tail(2).reset_index(drop=True))
# compare_two_length_width(sister_mmsis.iloc[[0, 2]].reset_index(drop=True))
def compare_three_length_width(sister_mmsis):
    # compare the length and width of the vessels pairwise among three rows
    if sister_mmsis.empty:
        return sister_mmsis
    if sister_mmsis.shape[0] != 3:
        return sister_mmsis
    sister_mmsis = sister_mmsis.sort_values(
            by=["s_lastPositionUpdate_timestamp"],
            ascending=[False],
        )   
    small_sister_mmsis = sister_mmsis[['mmsi',"s_lastPositionUpdate_timestamp", 's_staticData_dimensions_length', 's_staticData_dimensions_width']]
    # select the first two rows
    first_two = small_sister_mmsis.head(2).reset_index(drop=True)
    # select the last two rows
    last_two = small_sister_mmsis.tail(2).reset_index(drop=True)
    # select the first and the last rows
    first_last = small_sister_mmsis.iloc[[0, 2]].reset_index(drop=True)
    # compare the first two rows
    mmsi_cluster_first_two = (
        compare_two_length_width(first_two)
        .filter(["mmsi", "mergedlist_cluster"])
    )
    # compare the last two rows
    mmsi_cluster_last_two = (
        compare_two_length_width(last_two)
        .filter(["mmsi", "mergedlist_cluster"])
    )
    # compare the first and the last rows
    mmsi_cluster_first_last = (
        compare_two_length_width(first_last)
        .filter(["mmsi", "mergedlist_cluster"])
    )

    # Perform inner join on mmsi
    second_row_cluster = mmsi_cluster_first_two.merge(mmsi_cluster_last_two, on="mmsi", how="inner")
    first_row_cluster = mmsi_cluster_first_two.merge(mmsi_cluster_first_last, on="mmsi", how="inner")
    third_row_cluster = mmsi_cluster_last_two.merge(mmsi_cluster_first_last, on="mmsi", how="inner")
    # concat rows
    mmsi_cluster = pd.concat([second_row_cluster, first_row_cluster, third_row_cluster])
    # check to see if two columns are the same
    if mmsi_cluster["mergedlist_cluster_x"].equals(mmsi_cluster["mergedlist_cluster_y"]):
        mmsi_cluster = mmsi_cluster.drop(columns=["mergedlist_cluster_y"])
        mmsi_cluster = mmsi_cluster.rename(columns={"mergedlist_cluster_x": "mergedlist_cluster"})
    return sister_mmsis.merge(mmsi_cluster, on="mmsi", how="left")

#compare_three_length_width(sister_mmsis)

#sister_mmsis = pd.read_csv("sister_mmsis.csv")

def compare_four_length_width(sister_mmsis):
    # compare the length and width of the vessels pairwise among four rows
    if sister_mmsis.empty:
        return sister_mmsis
    if sister_mmsis.shape[0] != 4:
        return sister_mmsis
    sister_mmsis = sister_mmsis.sort_values(
            by=["s_lastPositionUpdate_timestamp"],
            ascending=[False],
        )   
    small_sister_mmsis = sister_mmsis[['mmsi',"s_lastPositionUpdate_timestamp", 's_staticData_dimensions_length', 's_staticData_dimensions_width']]
    # select the first two rows
    first_two = small_sister_mmsis.iloc[[0, 1]].reset_index(drop=True)
    # select the last two rows
    second_two = small_sister_mmsis.iloc[[2, 3]].reset_index(drop=True)
    mmsi_cluster_first_two = (
        compare_two_length_width(first_two)
        .filter(["mmsi", "mergedlist_cluster"])
    )
    mmsi_cluster_second_two = (
        compare_two_length_width(second_two)
        .filter(["mmsi", "mergedlist_cluster"])
    )
    # Perform inner join on mmsi
    combined = pd.concat([mmsi_cluster_first_two, mmsi_cluster_second_two]).reset_index(drop=True)
    # check to see if two columns are the same
    if combined["mergedlist_cluster"].nunique() == 1:
        return sister_mmsis.merge(combined, on="mmsi", how="left")
    return sister_mmsis


def get_same_vessels(sister_mmsis):

    if sister_mmsis.empty:
        return sister_mmsis

    if sister_mmsis.shape[0] == 2:
        return compare_two_length_width(sister_mmsis)
    
    if sister_mmsis.shape[0] == 3:
        return compare_three_length_width(sister_mmsis)
    
    if sister_mmsis.shape[0] == 4:
        return compare_four_length_width(sister_mmsis)

    numeric_features = [
        "s_staticData_dimensions_length",
        "s_staticData_dimensions_width",
    ]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = ["s_staticData_shipType"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    st.write(sister_mmsis.shape[0])

    num_rows = sister_mmsis.shape[0]

    if num_rows >= 10:
        n_clusters = 10
    elif 5 <= num_rows < 10:
        n_clusters = 3
    elif 2 <= num_rows < 5:
        n_clusters = 2
    elif num_rows < 2:
        n_clusters = 1 

    # Create a clustering pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("cluster", KMeans(n_clusters=n_clusters, max_iter=2, n_init=10, random_state=42)),
        ]
    )
    # Fit the pipeline
    pipeline.fit(sister_mmsis)

    # Predict clusters
    sister_mmsis["mergedlist_cluster"] = pipeline.predict(sister_mmsis)
    # order them by cluster and then by lastPositionUpdate_timestamp desc
    sister_mmsis = sister_mmsis.sort_values(
        by=["mergedlist_cluster", "s_lastPositionUpdate_timestamp"],
        ascending=[True, False],
    )
    return sister_mmsis

# check this later
def get_same_vessels_dbscan(sister_mmsis):
    
        if sister_mmsis.empty:
            return sister_mmsis
        
        numeric_features = [
            "s_staticData_dimensions_length",
            "s_staticData_dimensions_width",
        ]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    
        categorical_features = ["s_staticData_shipType"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
    
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        st.write(sister_mmsis.shape[0])

        # Create a clustering pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("cluster", DBSCAN(eps=0.00001, min_samples=2)),
            ]
        )
        # Fit the pipeline
        #pipeline.fit(sister_mmsis)
        sister_mmsis["mergedlist_cluster"] = pipeline.fit_predict(sister_mmsis)

        # Predict clusters
        #sister_mmsis["mergedlist_cluster"] = pipeline.predict(sister_mmsis)
        # order them by cluster and then by lastPositionUpdate_timestamp desc
        sister_mmsis = sister_mmsis.sort_values(
            by=["mergedlist_cluster", "s_lastPositionUpdate_timestamp"],
            ascending=[True, False],
        )
        # Predict clusters
        return sister_mmsis

def get_same_vessels_journey(journey_df):
    if journey_df.empty:
        return journey_df

    numeric_features = [
        "a",
        "b",
        "c",
        "d",
        "length",
        "width",
    ]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = ["shipBuilder", "vesselType"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # if number of samples is less than 10, we will use the number of samples as the number of clusters
    if journey_df.shape[0] < 10 and journey_df.shape[0] > 2:
        n_clusters = 3
    elif journey_df.shape[0] < 3 and journey_df.shape[0] > 1:
        n_clusters = 2
    elif journey_df.shape[0] == 1:
        n_clusters = 1
    else:
        n_clusters = 10
    # Create a clustering pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=2024)),
        ]
    )

    # Fit the pipeline
    pipeline.fit(journey_df)

    # Predict clusters
    journey_df["journey_cluster"] = pipeline.predict(journey_df)
    # order them by cluster and then by lastPositionUpdate_timestamp desc
    journey_df = journey_df.sort_values(
        by=["journey_cluster", "lastSysDate"], ascending=[True, False]
    )
    return journey_df


def stop_length_threshold(sister_mmsis, mmsi, threshold=70):
    mmsi = mmsi
    lenght_mmsi = (
        sister_mmsis
        .query("mmsi == @mmsi")
        .loc[:, "s_staticData_dimensions_length"]
    )
    if not lenght_mmsi.empty:
        lenght_mmsi = lenght_mmsi.values[0]
    else:
        lenght_mmsi = None
    if lenght_mmsi < threshold:
        #st.write("Vessel Length is less than 30 meters, not showing any data")
        return True
        #st.stop()
    elif lenght_mmsi is None:
        return True
        #st.stop()
    #st.write(f"Vessel Length is {lenght_mmsi}, that is greater than 30 meters, showing data")
    print(f"Vessel Length is {lenght_mmsi}, that is greater than {threshold} meters, showing data")
    return False
