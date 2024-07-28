import streamlit as st
import pandas as pd
from main import *
import requests
import os
import json
import dotenv

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Load the environment variables
# define enpoints
dotenv.load_dotenv()
AWS_MAGICPORT_SERVER_URL = os.environ.get("AWS_MAGICPORT_SERVER_URL")

vessels_merged_list_url = "http://" + AWS_MAGICPORT_SERVER_URL + "/Vessels/MergedList"
vessels_journey_url = "http://" + AWS_MAGICPORT_SERVER_URL + "/Vessels/Journey"
current_directory = os.getcwd()

# Set the page title
st.title("Vessel clustering")

# Create a numeric input field
imo_mmsi_name = st.selectbox("Select the type of the input:", ["IMO", "MMSI", "Name"])
if imo_mmsi_name == "IMO":
    imo = st.number_input("IMO:", value=8388608)
elif imo_mmsi_name == "MMSI":
    mmsi = st.number_input("MMSI:", value=352002954)
else:
    name = st.text_input("Name:", value="GUIGANGXINGTAI6398")
#mmsi = st.number_input("MMSI:", value=538007760)
submit = st.button("Query")



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
    mmsi = (
        pd.DataFrame(response.json())
        .rename(columns={"s_staticData_mmsi": "mmsi"})
        .query("s_lastPositionUpdate_timestamp.notnull()")
        .query("s_staticData_dimensions_length >= 30")
        .sort_values(by="s_lastPositionUpdate_timestamp", ascending=False)
        .head(1)
        .iloc[0]['mmsi']
    )
    return mmsi


def get_latest_mmsi_name(name):
    # fix filter below
    data = {
        "columns": [
            "s_staticData_mmsi",
            "s_lastPositionUpdate_timestamp",
            "s_staticData_dimensions_length",
        ],
        "s_staticData_name": name,
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
    mmsi = (
        pd.DataFrame(response.json())
        .rename(columns={"s_staticData_mmsi": "mmsi"})
        .query("s_lastPositionUpdate_timestamp.notnull()")
        .query("s_staticData_dimensions_length >= 30")
        .sort_values(by="s_lastPositionUpdate_timestamp", ascending=False)
        .head(1)
        .iloc[0]['mmsi']
    )
    return mmsi


def get_merged_list_sister_mmsis(mmsi, imo_given = False):
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


def get_same_vessels(sister_mmsis):

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

    # if number of samples is less than 10, we will use the number of samples as the number of clusters
    if sister_mmsis.shape[0] < 10 and sister_mmsis.shape[0] > 2:
        n_clusters = 3
    elif sister_mmsis.shape[0] < 3 and sister_mmsis.shape[0] > 1:
        n_clusters = 2
    elif sister_mmsis.shape[0] == 1:
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
    pipeline.fit(sister_mmsis)

    # Predict clusters
    sister_mmsis["mergedlist_cluster"] = pipeline.predict(sister_mmsis)
    # order them by cluster and then by lastPositionUpdate_timestamp desc
    sister_mmsis = sister_mmsis.sort_values(
        by=["mergedlist_cluster", "s_lastPositionUpdate_timestamp"],
        ascending=[True, False],
    )
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


def stop_length_threshold(sister_mmsis, mmsi, threshold=30):
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
    print(f"Vessel Length is {lenght_mmsi}, that is greater than 30 meters, showing data")
    return False

if submit:
    # get the imo first
    if imo_mmsi_name == "IMO":
        mmsi = get_latest_mmsi(imo)
        (corresponding_imo_code, sister_mmsis) = get_merged_list_sister_mmsis(mmsi, imo_given=True)
    elif imo_mmsi_name == "MMSI":
        (corresponding_imo_code, sister_mmsis) = get_merged_list_sister_mmsis(mmsi)
    elif imo_mmsi_name == "Name":
        st.write("Name is not implemented yet")
        st.stop()
    # lentgh threshold
    if stop_length_threshold(sister_mmsis, mmsi):
        st.write("Vessel Length is less than 30 meters, not showing any data")
        st.stop()
    sister_mmsis = (
        sister_mmsis
        .query("s_staticData_dimensions_length >= 30")
    )
    sister_mmsi_list = sister_mmsis["mmsi"].tolist()
    sister_mmsis = get_same_vessels(sister_mmsis)


        # i would like to grouo by mergedlist_cluster and display the df for each cluster separately in a column
    # st.cols should depend on the number of clusters

    number_of_clusters = sister_mmsis["mergedlist_cluster"].nunique()
    corresponding_mergedlist_cluster = (
        sister_mmsis
        .query("mmsi == @mmsi")
        .loc[:, "mergedlist_cluster"]
        .values[0]
    )
    corresponding_mergedlist_cluster = sister_mmsis[sister_mmsis["mmsi"] == mmsi][
        "mergedlist_cluster"
    ].values[0]

    sister_mmsis_in_same_cluster = sister_mmsis[
        sister_mmsis["mergedlist_cluster"] == corresponding_mergedlist_cluster
    ]
    different_vessels_counts = (
        sister_mmsis.groupby("mergedlist_cluster").size().reset_index()
    )
    different_vessels_counts.columns = ["mergedlist_cluster", "count"]
    sister_mmsi_list_in_same_cluster = sister_mmsis_in_same_cluster["mmsi"].tolist()
    if imo_mmsi_name == "IMO":
        st.write("Inferred MMSI: ", mmsi)
    elif imo_mmsi_name == "MMSI":
        st.write("Corresponding IMO Code: ", corresponding_imo_code)
    st.write("Sister MMSIs: ", sister_mmsis)
    # st.write("Different Vessels Counts: ", different_vessels_counts)

    # show ALL the clusters
    # st.write("Number of Clusters: ", number_of_clusters)
    # for i in range(number_of_clusters):
    #     group_mmis = sister_mmsis[sister_mmsis["mergedlist_cluster"] == i]
    #     st.write(group_mmis)


    st.write("Sister MMSIs in the same cluster: ", sister_mmsis_in_same_cluster)

    # WE EXPECT THAT BELOW list has only one element

    core_vessel_where_clause = f"imo_code = {corresponding_imo_code}"
    # cetificates_where_clause = f"mi.imo = {corresponding_imo_code}"
    if len(sister_mmsi_list_in_same_cluster) == 1:
        cetificates_where_clause = f"mi.mmsi = {sister_mmsi_list_in_same_cluster[0]}"
    else:
        cetificates_where_clause = (
            f"mi.mmsi in {tuple(sister_mmsi_list_in_same_cluster)}"
        )

    # # get vessel profile info: Vessel + Management
    # query = f"""
    #     WITH corresponding_imo_code AS (
    #         SELECT cv.imo_code
    #         FROM magicportai_core.core_vessels cv
    #         WHERE mmsi = {mmsi}
    #     )
    #     SELECT cv.*, evm.*
    #     FROM magicportai_core.core_vessels cv
    #     LEFT JOIN magicportai_crawl.equasis_vessels ev ON cv.mmsi = ev.mmsi
    #     LEFT JOIN magicportai_crawl.equasis_vessel_managements evm ON ev.id = evm.equasis_vessel_id
    #     JOIN corresponding_imo_code tic ON cv.imo_code = tic.imo_code
    # """

    # # vessel_profile_info_Vessels_Management = get_magicport_response(
    # #     database_location="liquidweb",
    # #     database_name="magicportai_crawl",
    # #     query=query,
    # #     prod=True,
    # # )
    if current_directory.startswith("/Users"):
        vessels_management_df = get_magicport_response(
            database_location="liquidweb",
            database_name="magicportai_crawl",
            query=f"""
                    select evm.*
                    from magicportai_crawl.equasis_vessels ev 
                    left join magicportai_crawl.equasis_vessel_managements evm on ev.id = evm.equasis_vessel_id 
                    where ev.imo = {corresponding_imo_code}
                """,
            prod=True,
        )
    else:
        vessels_management_df = pd.DataFrame({"Message:": ["Not brining data due to security reasons"]})
    # else:
    #     filters = [('imo', '==', corresponding_imo_code)]
    #     vessels_management_df = (
    #         pd.read_parquet("data/vessels_managemnet", filters = filters, engine='fastparquet')
    #         .query("imo == @corresponding_imo_code")
    #     )
            # cerate a dataframe from the response

    # journey_df_imo = get_journey_data(
    #     [corresponding_imo_code], "2024-01-22 10:00:00", "2024-07-22 10:00:00"
    # )
    # journey_df_imo = get_same_vessels_journey(journey_df_imo)
    # journey_df_mmsi = get_journey_data(
    #     sister_mmsi_list, "2024-01-22 10:00:00", "2024-07-22 10:00:00", imo=False
    # )
    # journey_df_mmsi = get_same_vessels_journey(journey_df_mmsi)
    journey_df_mmsi_in_same_cluster = get_journey_data(
        sister_mmsi_list_in_same_cluster,
        "2024-01-22 10:00:00",
        "2024-07-22 10:00:00",
        imo=False,
    )
    journey_df_mmsi_in_same_cluster = get_same_vessels_journey(
        journey_df_mmsi_in_same_cluster
    )
    # merged_list_df = get_merged_list_data([corresponding_imo_code])
    merged_list_df_mmsi_in_same_cluster = get_merged_list_data(
        sister_mmsi_list_in_same_cluster, imo=False
    )
    
    if current_directory.startswith("/Users"):
        vessel_certificate_df = get_magicport_response(
            database_location="liquidweb",
            database_name="magicportai_mou",
            query="""
                    select mic.*, mi.*
                    from mou_inspection_certificates mic 
                    left join mou_inspections mi on mic.mou_inspection_id = mi.id"""
            + f" where {cetificates_where_clause}",
            prod=True,
        )
    else:
        vessel_certificate_df = pd.DataFrame({"Message:": ["Not brining data due to security reasons"]})

    # else:
    #     if len(sister_mmsi_list_in_same_cluster) == 1:
    #         filters = [('mmsi', '==', sister_mmsi_list_in_same_cluster[0])]
    #         query = "mmsi == @sister_mmsi_list_in_same_cluster[0]"
    #     else:
    #         filters = [('mmsi', 'in', {tuple(sister_mmsi_list_in_same_cluster)})]
    #         query = "mmsi in @sister_mmsi_list_in_same_cluster"
    #     vessel_certificate_df = (
    #         pd.read_parquet("data/vessel_certificate", filters = filters, engine='fastparquet')
    #         .query(query)
    #     )

    # core_vessels_df = get_magicport_response(
    #     database_location="liquidweb",
    #     database_name="magicportai_core",
    #     query=f"""
    #             select *
    #             from magicportai_core.core_vessels
    #             where {core_vessel_where_clause}
    #             order by s_lastPositionUpdate_updateTimestamp desc
    #         """,
    #     prod=True,
    # )

    st.write("management")
    management_box_columns = ["imo", "role", "company", "address"]
    st.write(vessels_management_df)
    # st.write("journey with imo")
    # st.write(journey_df_imo)
    # st.write("journey with mmsi")
    # st.write(journey_df_mmsi)
    st.write("journey with mmsi in the same cluster")
    st.write(journey_df_mmsi_in_same_cluster)
    # st.write("merged list")
    # st.write(merged_list_df)
    st.write("merged list with mmsi in the same cluster")
    st.write(merged_list_df_mmsi_in_same_cluster)
    st.write("certificates with mmsi in the same cluster")
    st.write(vessel_certificate_df)
    vessel_box_columns = [
        "mmsi",
        "imo_code",
        "name",
        "callsign_code",
        "sub_type",
        "vessel_type",
        "gross_tonnage",
        "net_tonnage",
        "dwt",
        "length",
        "built_year",
        "ship_builder",
    ]
