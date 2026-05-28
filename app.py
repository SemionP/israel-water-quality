
[     UTC     ] Logs for israel-water-quality.streamlit.app/
────────────────────────────────────────────────────────────────────────────────────────
[22:40:34] 🚀 Starting up repository: 'israel-water-quality', branch: 'main', main module: 'app.py'
[22:40:34] 🐙 Cloning repository...
[22:40:35] 🐙 Cloning into '/mount/src/israel-water-quality'...

[22:40:35] 🐙 Cloned repository!
[22:40:35] 🐙 Pulling code changes from Github...
[22:40:35] 📦 Processing dependencies...

──────────────────────────────────────── uv ───────────────────────────────────────────

Using uv pip install.
Using Python 3.14.5 environment at /home/adminuser/venv
Resolved 74 packages in 699ms
Prepared 74 packages in 2.26s
Installed 74 packages in 102ms
 + altair==6.1.0
 + annotated-types==0.7.0
 + anyio==4.13.0
 + attrs==26.1.0
 + blinker==1.9.0
 + branca==0.8.2
 + cachetools==7.1.4
 + certifi==2026.5.20
 [2026-05-27 22:40:38.767716] + cffi==2.0.0
 + charset-normalizer==3.4.7
 + click==8.4.1
 [2026-05-27 22:40:38.767925] + cryptography==48.0.0
 + distro==1.9.0
 + earthengine-api==1.7.28
 + folium==0.20.0
 + gitdb==4.0.12
 + [2026-05-27 22:40:38.768154] gitpython==3.1.50
 + google-api-core==2.30.3
 + google-api-python-client==2.196.0
 + google-auth==2.53.0
 + google-auth-httplib2==0.4.0
 + google-cloud-core==2.6.0[2026-05-27 22:40:38.768397] 
 + google-cloud-storage==3.10.1
 + google-crc32c==1.8.0
 + google-genai==2.6.0
 + google-resumable-media==2.9.0
 + [2026-05-27 22:40:38.768677] googleapis-common-protos==1.75.0
 + h11==0.16.0
 + httpcore==1.0.9
 + httplib2==0.31.2
 + httptools==[2026-05-27 22:40:38.768908] 0.8.0[2026-05-27 22:40:38.769159] 
 + httpx==0.28.1
 + idna==3.16
 + itsdangerous==[2026-05-27 22:40:38.769418] 2.2.0
 + jinja2==3.1.6
 + jsonschema==4.26.0
 + jsonschema-specifications==2025.9.1
 +[2026-05-27 22:40:38.769598]  markupsafe==3.0.3
 + narwhals==2.21.2
 + numpy==2.4.6
 + packaging==26.2
 + [2026-05-27 22:40:38.769800] pandas==3.0.3
 + pillow==12.2.0
 + proto-plus==1.28.0
 + protobuf==7.35.0
 + pyarrow[2026-05-27 22:40:38.769941] ==24.0.0
 + pyasn1==0.6.3
 + pyasn1-modules==0.4.2
 + pycparser==3.0
 + pydantic==2.13.4[2026-05-27 22:40:38.770087] 
 + pydantic-core==2.46.4
 + pydeck==0.9.2
 + pyparsing==3.3.2
 + python-dateutil==2.9.0.post0[2026-05-27 22:40:38.770208] 
 + python-multipart==0.0.29
 + referencing==0.37.0
 + requests==2.34.2
 + rpds-py==0.30.0
 + six==1.17.0
 + smmap==5.0.3
 + sniffio==1.3.1
 + starlette==1.1.0
 + streamlit==1.57.0
 + [2026-05-27 22:40:38.770570] streamlit-folium==0.27.2
 + tenacity==9.1.4
 + toml==0.10.2
 + typing-extensions==4.15.0
 [2026-05-27 22:40:38.770880] + typing-inspection==0.4.2
 + uritemplate==4.2.0
 + urllib3==2.7.0
 [2026-05-27 22:40:38.771015] + uvicorn==0.48.0
 + watchdog==6.0.0[2026-05-27 22:40:38.771146] 
 + websockets==16.0
 + xyzservices==2026.3.0
Checking if Streamlit is installed
Found Streamlit version 1.57.0 in the environment
Installing rich for an improved exception logging
Using uv pip install.
Using Python 3.14.5 environment at /home/adminuser/venv
Resolved 4 packages in 123ms
Prepared 4 packages in 105ms
Installed 4 packages in 10ms
 + markdown-it-py==4.2.0
 + mdurl[2026-05-27 22:40:40.366507] ==0.1.2
 + pygments==2.20.0
 + rich==15.0.0

────────────────────────────────────────────────────────────────────────────────────────

[22:40:41] 🐍 Python dependencies were installed from /mount/src/israel-water-quality/requirements.txt using uv.
Check if streamlit is installed
Streamlit is already installed
[22:40:42] 📦 Processed dependencies!
2026-05-27 22:40:44.224 Uvicorn server started on 0.0.0.0:8501



2026-05-27 22:41:08.319 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-27 22:41:08.319 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
/mount/src/israel-water-quality/app.py:328: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  end=datetime.utcnow(); start=end-timedelta(days=days_back)
/mount/src/israel-water-quality/app.py:332: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  return sorted(list(set([datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in dl])),reverse=True)
/mount/src/israel-water-quality/app.py:389: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  img_dt = datetime.utcfromtimestamp(img_time_ms / 1000)
/mount/src/israel-water-quality/app.py:390: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  age_hours = (datetime.utcnow() - img_dt).total_seconds() / 3600
2026-05-27 22:41:11.339 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
/mount/src/israel-water-quality/app.py:424: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  age_h      = (datetime.utcnow() - datetime.utcfromtimestamp(img_time/1000)).total_seconds() / 3600
/mount/src/israel-water-quality/app.py:424: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  age_h      = (datetime.utcnow() - datetime.utcfromtimestamp(img_time/1000)).total_seconds() / 3600
[22:43:25] 🐙 Pulling code changes from Github...
[22:43:26] 📦 Processing dependencies...
[22:43:26] 📦 Processed dependencies!
2026-05-27 22:43:26.104 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-27 22:43:27.307 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
[22:43:27] 🔄 Updated app!
2026-05-27 22:43:29.103 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'` or specify an integer width.
ConnectionClosedError exception in shielded future
future: <Future finished exception=ConnectionClosedError(None, Close(code=<CloseCode.INTERNAL_ERROR: 1011>, reason='keepalive ping timeout'), None)>
Traceback (most recent call last):
  File "/home/adminuser/venv/lib/python3.14/site-packages/websockets/legacy/protocol.py", line 1276, in close_connection
    await self.transfer_data_task
  File "/home/adminuser/venv/lib/python3.14/site-packages/websockets/legacy/protocol.py", line 940, in transfer_data
    message = await self.read_message()
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/adminuser/venv/lib/python3.14/site-packages/websockets/legacy/protocol.py", line 1010, in read_message
    frame = await self.read_data_frame(max_size=self.max_size)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/adminuser/venv/lib/python3.14/site-packages/websockets/legacy/protocol.py", line 1087, in read_data_frame
    frame = await self.read_frame(max_size)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/adminuser/venv/lib/python3.14/site-packages/websockets/legacy/protocol.py", line 1144, in read_frame
    frame = await Frame.read(
            ^^^^^^^^^^^^^^^^^
    ...<4 lines>...
    )
    ^
  File "/home/adminuser/venv/lib/python3.14/site-packages/websockets/legacy/framing.py", line 70, in read
    data = await reader(2)
           ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.14/asyncio/streams.py", line 769, in readexactly
    await self._wait_for_data('readexactly')
  File "/usr/local/lib/python3.14/asyncio/streams.py", line 539, in _wait_for_data
    await self._waiter
asyncio.exceptions.CancelledError

The above exception was the direct cause of the following exception:

websockets.exceptions.ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout; no close frame received
2026-05-28 04:26:54.511 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-28 04:26:54.511 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
/mount/src/israel-water-quality/app.py:328: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  end=datetime.utcnow(); start=end-timedelta(days=days_back)
/mount/src/israel-water-quality/app.py:332: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  return sorted(list(set([datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in dl])),reverse=True)
/mount/src/israel-water-quality/app.py:389: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  img_dt = datetime.utcfromtimestamp(img_time_ms / 1000)
/mount/src/israel-water-quality/app.py:390: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  age_hours = (datetime.utcnow() - img_dt).total_seconds() / 3600
2026-05-28 04:26:57.464 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
/mount/src/israel-water-quality/app.py:424: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  age_h      = (datetime.utcnow() - datetime.utcfromtimestamp(img_time/1000)).total_seconds() / 3600
/mount/src/israel-water-quality/app.py:424: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  age_h      = (datetime.utcnow() - datetime.utcfromtimestamp(img_time/1000)).total_seconds() / 3600
2026-05-28 04:27:03.756 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'` or specify an integer width.
2026-05-28 06:04:11.617 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-28 06:04:11.618 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-28 06:04:12.789 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2026-05-28 06:04:15.476 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'` or specify an integer width.
[06:23:03] 🐙 Pulling code changes from Github...
[06:23:04] 📦 Processing dependencies...
[06:23:04] 📦 Processed dependencies!
[06:23:05] 🔄 Updated app!
2026-05-28 06:23:06.492 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-28 06:23:06.493 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
/mount/src/israel-water-quality/app.py:328: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  end   = datetime.utcnow()
/mount/src/israel-water-quality/app.py:337: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in dl
/mount/src/israel-water-quality/app.py:396: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  img_dt = datetime.utcfromtimestamp(img_time_ms / 1000)
/mount/src/israel-water-quality/app.py:397: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  age_hours = (datetime.utcnow() - img_dt).total_seconds() / 3600
2026-05-28 06:23:08.623 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2026-05-28 06:23:09.381 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'` or specify an integer width.
2026-05-28 06:23:12.481 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-28 06:23:12.482 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-28 06:23:12.854 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2026-05-28 06:23:13.501 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'` or specify an integer width.
2026-05-28 06:23:15.393 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-28 06:23:15.393 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
2026-05-28 06:23:15.816 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2026-05-28 06:23:16.511 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'` or specify an integer width.
[06:30:09] 🐙 Pulling code changes from Github...
[06:30:10] 📦 Processing dependencies...
[06:30:10] 📦 Processed dependencies!
2026-05-28 06:30:10.099 Please replace `st.components.v1.html` with `st.iframe`.

`st.components.v1.html` will be removed after 2026-06-01.
/mount/src/israel-water-quality/app.py:442: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  end   = datetime.utcnow()
/mount/src/israel-water-quality/app.py:452: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  s3_dates = set(datetime.utcfromtimestamp(d/1000).strftime(date_fmt) for d in s3_ts)
/mount/src/israel-water-quality/app.py:459: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  mod_dates = set(datetime.utcfromtimestamp(d/1000).strftime(date_fmt) for d in mod_ts)
2026-05-28 06:30:10.865 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
[06:30:11] 🔄 Updated app!
/mount/src/israel-water-quality/app.py:517: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  age_h      = (datetime.utcnow() - datetime.utcfromtimestamp(img_time/1000)).total_seconds() / 3600
/mount/src/israel-water-quality/app.py:517: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
  age_h      = (datetime.utcnow() - datetime.utcfromtimestamp(img_time/1000)).total_seconds() / 3600
2026-05-28 06:30:13.370 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'` or specify an integer width.
[06:30:44] 🔌 Disconnecting...
[06:30:49] 🖥 Provisioning machine...
[06:30:44] 🚀 Starting up repository: 'israel-water-quality', branch: 'main', main module: 'app.py'
[06:30:44] 🐙 Cloning repository...
[06:30:45] 🐙 Cloning into '/mount/src/israel-water-quality'...

[06:30:45] 🐙 Cloned repository!
[06:30:45] 🐙 Pulling code changes from Github...
[06:30:45] 📦 Processing dependencies...

──────────────────────────────────────── uv ───────────────────────────────────────────

Using uv pip install.
Using Python 3.14.5 environment at /home/adminuser/venv
Resolved 74 packages in 687ms
Prepared 74 packages in 2.17s
[06:30:50] 🎛 Preparing system...
[06:30:50] ⛓ Spinning up manager process...
Installed 74 packages in 383ms
 + altair==6.1.0
 + annotated-types==0.7.0
 + anyio==4.13.0
 + attrs==26.1.0
 + blinker==1.9.0
 + branca==0.8.2[2026-05-28 06:30:48.978133] 
 + cachetools==7.1.4
 + certifi==2026.5.20
 + cffi==2.0.0
 + charset-normalizer==3.4.7
 + click==8.4.1
 + cryptography==48.0.0
 + [2026-05-28 06:30:48.978359] distro==1.9.0
 + earthengine-api==1.7.28
 + folium==0.20.0
 + gitdb==4.0.12
 + gitpython==3.1.50
 + google-api-core==2.30.3
 + google-api-python-client==2.196.0
 + google-auth==2.53.0
 + google-auth-httplib2==0.4.0
 + google-cloud-core==2.6.0[2026-05-28 06:30:48.978649] 
 + google-cloud-storage==3.10.1
 + google-crc32c==1.8.0
 + google-genai==2.6.0
 + google-resumable-media==2.9.0
 + googleapis-common-protos==1.75.0
 + h11==0.16.0
 [2026-05-28 06:30:48.978906] + httpcore==1.0.9
 + httplib2==0.31.2
 + httptools==0.8.0
 + httpx==0.28.1
 + idna==3.16
 + itsdangerous==2.2.0
 + jinja2==3.1.6
 + jsonschema==4.26.0[2026-05-28 06:30:48.979302] 
 + jsonschema-specifications==2025.9.1
 + markupsafe==3.0.3
 + narwhals==2.21.2[2026-05-28 06:30:48.979494] 
 + numpy==2.4.6
 + packaging==26.2
 + pandas==3.0.3
 + pillow==[2026-05-28 06:30:48.979703] 12.2.0
 + proto-plus==1.28.0
 + protobuf==7.35.0
 + [2026-05-28 06:30:48.979921] pyarrow==24.0.0
 + pyasn1==0.6.3
 + pyasn1-modules==0.4.2
 + pycparser==3.0
 + pydantic==2.13.4
 + pydantic-core==2.46.4[2026-05-28 06:30:48.980170] 
 + pydeck==0.9.2
 + pyparsing==3.3.2
 + python-dateutil==2.9.0.post0
 + python-multipart[2026-05-28 06:30:48.980272] ==0.0.29
 + referencing==0.37.0
 + requests==2.34.2
 + rpds-py==0.30.0[2026-05-28 06:30:48.980373] 
 + six==1.17.0
 + smmap==5.0.3
 + sniffio==1.3.1
 + starlette[2026-05-28 06:30:48.980524] ==1.1.0
 + streamlit==1.57.0
 + streamlit-folium==0.27.2
 + tenacity==9.1.4
 + [2026-05-28 06:30:48.980669] toml==0.10.2
 + typing-extensions==4.15.0
 + typing-inspection==0.4.2
 + uritemplate[2026-05-28 06:30:48.980838] ==4.2.0
 + urllib3==2.7.0
 + uvicorn==0.48.0
 + watchdog==[2026-05-28 06:30:48.980966] 6.0.0
 + websockets==16.0
 + xyzservices==2026.3.0
Checking if Streamlit is installed
Found Streamlit version 1.57.0 in the environment
Installing rich for an improved exception logging
Using uv pip install.
Using Python 3.14.5 environment at /home/adminuser/venv
Resolved 4 packages in 120ms
Prepared 4 packages in 123ms
Installed 4 packages in 11ms
 + markdown-it-py==4.2.0
 + mdurl==0.1.2
 + pygments==2.20.0
 [2026-05-28 06:30:50.737804] + rich==15.0.0

────────────────────────────────────────────────────────────────────────────────────────

[06:30:51] 🐍 Python dependencies were installed from /mount/src/israel-water-quality/requirements.txt using uv.
Check if streamlit is installed
Streamlit is already installed
[06:30:52] 📦 Processed dependencies!
2026-05-28 06:30:54.247 Uvicorn server started on 0.0.0.0:8501


