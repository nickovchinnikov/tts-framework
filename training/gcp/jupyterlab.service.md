1. Create a new service file:

```
sudo nano /etc/systemd/system/jupyterlab.service
```

2. In the editor, add the following content:

```
[Unit]
Description=JupyterLab

[Service]
Type=simple
ExecStart=/home/you/anaconda3/bin/jupyter lab --ip=0.0.0.0 --no-browser --port=8888 --NotebookApp.token='93058db2c647621fb48974ac8902c20b'
User=you
WorkingDirectory=/home/you
Restart=always

[Install]
WantedBy=multi-user.target
```

If any trouble with the env arise, add to the service file:

```
Environment=PATH=/home/you/anaconda3/envs/tts_framework/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

3. Save the file and exit the editor.

4. Enable the service to start on boot:

```
sudo systemctl enable jupyterlab
```

5. Start the service:

```
sudo systemctl start jupyterlab
```

6. In case of any fixes of the service file, reload the daemons

```
sudo systemctl daemon-reload
```
