## Requirements
- Python 3.7 or newer
- pip
- virtualenv
- Docker

```bash
# This needs to be done ONCE:
docker pull djstrong/krnnt:1.0.1
# To run:
docker run -p 9003:9003 -it djstrong/krnnt
# To kill, ctrl+c
```

In another terminal window:

```bash
# This needs to be done ONCE:
virtualenv
source bin/activate
pip3 install -r requirements.txt # this may be just pip on some platforms
deactivate
# To run:
source bin/activate
python3 run.py
# After you're done:
deactivate
```
