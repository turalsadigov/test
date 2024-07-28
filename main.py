import pandas as pd
import sshtunnel
import time
import pymysql
from dotenv import load_dotenv
load_dotenv()


import os
import tempfile
import paramiko

# Retrieve the private key from environment variable
private_key_content = os.environ.get("MAGICPORT_DATABASE_ACCESS_KEY")
if private_key_content is None:
    raise ValueError("Environment variable PRIVATE_KEY is not set")

# Get current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Write the private key content to a temporary file
# with tempfile.NamedTemporaryFile(dir='.', delete=False) as temp_pem:
#     temp_pem.write(private_key_content.encode())
#     temp_pem_path = temp_pem.name

try:
    with tempfile.NamedTemporaryFile(dir=current_dir, delete=False) as temp_pem:
        temp_pem.write(private_key_content.encode())
        temp_pem_path = temp_pem.name
    print(f"Temporary file created at: {temp_pem_path}")
    # print the content of the temporary file
    with open(temp_pem_path, "r") as f:
        print("reading temp file: ",f.read())
except Exception as e:
    print(f"Error creating temporary file: {e}")


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
            ("ec2-54-191-232-27.us-west-2.compute.amazonaws.com", 22),
            ssh_username="ec2-user",
            ssh_pkey=temp_pem_path,
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
                df = pd.DataFrame({"MySQL Error:": [err]})
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
