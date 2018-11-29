# Demo of a Self-Driving Car Steering Model for Clusterone
<p align="center">
<img src="co_logo.png" alt="Clusterone" width="200">
</p>

This is a basic self-steering car model implemented in TensorFlow. It is used as a demo project to get started with the [Clusterone](https://clusterone.com) deep learning computation platform.

Follow the [Getting Started guide](https://docs.clusterone.com/get-started) for Clusterone and read the author's original blog post about building this demo [here](https://clusterone.com/tutorials/self-steering-car-in-tensorflow).


## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [License](#license)

## Install

To run this project, you need:

- [Python](https://python.org) 3.5+
- [Git](https://git-scm.com/)
- The Clusterone Python library. Install it with `pip install clusterone`
- GitHub account. Create an account for free on [https://github.com/](https://github.com/).
- A Clusterone account. Sign up [here](https://clusterone.com/join-waitlist/)  for your SaaS account.

## Usage

Running a job on Clusterone is simple with the `just` command line tool that comes included with the Clusterone Python package.

### Setting up

Follow the **Set Up** section of the [Get Started](https://docs.clusterone.com/get-started#set-up) guide to add your GitHub personal access token to your Clusterone account.

Then follow [Create a project](https://docs.clusterone.com/get-started#create-a-project) section to add self-driving-demo project. Use **`clusterone/self-driving-demo`** repository instead of what is shown in the guide.

### Run on Clusterone

These instructions use the `just` command line tool. It comes with the Clusterone Python library and is installed automatically with the library.

If you have used Clusterone library before with a different Clusterone installation, make sure it is connected to the correct endpoint by running `just config endpoint https://clusterone.com`.

Log into your Clusterone account using `just login`, and entering your login information.

First, let's make sure that you have the project. Execute the command `just get projects` to see all your projects. You should see something like this:
```shell
>> just get projects
All projects:

| # | Project                    | Created at          | Description |
|---|----------------------------|---------------------|-------------|
| 0 | username/self-driving-demo | 2018-11-19T21:13:30 |             |
```
where `username` should be your Clusterone account name.

Let's create a job. Make sure to replace `username` with your username.
```shell
just create job distributed \
  --project username/self-driving-demo \
  --name sdc-first-job \
  --docker-image tensorflow-1.11.0-cpu-py35 \
  --ps-docker-image tensorflow-1.11.0-cpu-py35 \
  --time-limit 1h \
  --command "python -m main --absolute_data_path /public/self-driving-demo-data/" \
  --setup-command "pip install -r requirements.txt"
```

Now the final step is to start the job:
```shell
just start job demo-user/sdc-first-job
```
you can find the `job name` by running the following:
```shell
>> just get jobs
All jobs:
| # | Name                      | Id       | Project                                | Status   | Launched at      |
|---|---------------------------|----------|----------------------------------------|----------|------------------|
| 1 | demo-user/sdc-first-job   | <JOBID>  | None/self-driving-demo:<PROJECTID>     | created  |                  |
```

You can monitor the execution of your job on Clusterone using `just get events`.

Instead of running the model from the command line, you can also use Clusterone's graphical web interface [Matrix](https://clusterone.com/matrix).

For a more detailed guide on how to run this project on Clusterone, check out the [Getting Started guide](https://docs.clusterone.com/get-started). To learn more about Clusterone, visit our [website](https://clusterone.com).

If you have any further questions, don't hesitate to write us a support ticket on [Clusterone.com](https://clusterone.com) or join us on [Slack](https://bit.ly/2OPc6JH)!

## License

[MIT](LICENSE) © Clusterone Inc.

Comma dataset and [data_reader.py](utils/data_reader.py) by [comma.ai](https://github.com/commaai/research), licensed as [specified](LICENSE_COMMA).
