# Demo of a Self-Driving Car Steering Model for Clusterone

This is a basic self-steering car model implemented in TensorFlow. It is used as a demo project to get started with the [Clusterone](https://clusterone.com) deep learning computation platform.


Follow the [Getting Started guide](https://docs.clusterone.com/get-started) for Clusterone and read the author's blog post about building this demo [here](https://clusterone.com/blog/2017/08/07/self-driving-car-tensorflow/).


## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [License](#license)

## Install

To run this project, you need:

- [Python](https://python.org) 3.5+
- [Git](https://git-scm.com/)
- The TensorFlow Python library. Get it with `pip install tensorflow`
- The Clusterone Python library. Install it with `pip install clusterone`
- A Clusterone account. [Join the waitlist](https://clusterone.com/join-waitlist/) if you don't have an account yet.

## Usage

Running a job on Clusterone is simple with the `just` command line tool that comes included with the Clusterone Python package.

### Run on Clusterone

To run the model on Clusterone, you first need a Clusterone account. Log in with `just login`.

`cd` into the directory where you cloned this repository to and create a new project with `just init project self-driving-demo`. The data is already uploaded to Clusterone, so you don't need to worry about it.

Push the project code to Clusterone with `git push clusterone master`.

When the upload is complete, create a job to run the model on Clusterone:

```bash
just create job distributed \
  --project self-driving-demo \
  --name sdc-first-job \
  --datasets /public/self-driving-demo-data \
  --docker-image tensorflow-1.11.0-cpu-py35 \
  --ps-docker-image tensorflow-1.11.0-cpu-py35 \
  --time-limit 1h \
  --command "python -m main" \
  --setup_command "pip install -r requirements.txt"
```

Now the final step is to start the job:

```bash
just start job -p self-driving-demo/sdc-first-job
```

You can monitor the execution of your job on Clusterone using `just get events`.

Instead of running the model from the command line, you can also use Clusterone's graphical web interface [Matrix](https://clusterone.com/matrix).

For a more detailed guide on how to run this project on Clusterone, check out the [Getting Started guide](https://docs.clusterone.com/get-started). To learn more about Clusterone, visit our [website](https://clusterone.com).

If you have any further questions, don't hesitate to write us a support ticket on [Clusterone.com](https://clusterone.com) or join us on [Slack](https://bit.ly/2OPc6JH)!

## License

[MIT](LICENSE) © Clusterone Inc.

Comma dataset and [data_reader.py](utils/data_reader.py) by [comma.ai](https://github.com/commaai/research), licensed as [specified](LICENSE_COMMA).
