import streamlit as st
import pandas as pd
from main import *
import os
import dotenv
from datetime import datetime, timedelta



# Load the environment variables
# define enpoints
dotenv.load_dotenv()
AWS_MAGICPORT_SERVER_URL = os.environ.get("AWS_MAGICPORT_SERVER_URL")

vessels_merged_list_url = "http://" + AWS_MAGICPORT_SERVER_URL + "/Vessels/MergedList"
vessels_journey_url = "http://" + AWS_MAGICPORT_SERVER_URL + "/Vessels/Journey"
current_directory = os.getcwd()
length_threshold = 70

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


if submit:
    # get the imo first
    if imo_mmsi_name == "IMO":
        mmsi = get_latest_mmsi(imo)
        if mmsi is None:
            st.write("Vessel Length is less than chosen threshold for all potential MMSIs, not showing any data")
            st.stop()
        (corresponding_imo_code, sister_mmsis) = get_merged_list_sister_mmsis(mmsi, imo_given=True, imo=imo)
    elif imo_mmsi_name == "MMSI":
        (corresponding_imo_code, sister_mmsis) = get_merged_list_sister_mmsis(mmsi)
    elif imo_mmsi_name == "Name":
        st.stop()
        # FIX THIS
        names_mmiss= get_names_mmsis(imo_mmsi_name)
        chosen_name = names_mmiss.head(1)["name"].values[0]
        mmsi = names_mmiss.head(1)["mmsi"].values[0]
        (corresponding_imo_code, sister_mmsis) = get_merged_list_sister_mmsis(mmsi)

        # st.write("Name is not implemented yet")
        # st.stop()
    # lentgh threshold
    if stop_length_threshold(sister_mmsis, mmsi):
        st.write(f"Vessel Length is less than chosen threshold, {length_threshold}, not showing any data")
        st.stop()
    sister_mmsis = (
        sister_mmsis
        .query("s_staticData_dimensions_length.notnull()")
        .query("s_staticData_dimensions_length >= 50") # 
    )
    #sister_mmsis.to_csv("sister_mmsis.csv", index=False)
    #st.write(sister_mmsis)
    sister_mmsi_list = sister_mmsis["mmsi"].tolist()
    sister_mmsis = get_same_vessels(sister_mmsis)
    #sister_mmsis = get_same_vessels_dbscan(sister_mmsis)


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
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # go back to 3 months ago
    three_months_ago = datetime.now() - timedelta(days=90)
    journey_df_mmsi_in_same_cluster = get_journey_data(
        sister_mmsi_list_in_same_cluster,
        three_months_ago.strftime("%Y-%m-%d %H:%M:%S"),
        current_date,
        imo=False,
    )
    # journey_df_mmsi_in_same_cluster = get_same_vessels_journey(
    #     journey_df_mmsi_in_same_cluster
    # )
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
