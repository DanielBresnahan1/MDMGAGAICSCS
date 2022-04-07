Local Tests are designed to run on the locally hosted flask app, and require the flask app to be running at localhost:5000
Real Tests are designed to run on the app hosted off of heroku.

If a file doesn't specify whether local or real, the file will run locally. 

These tests are selenium tests, which test the front end.