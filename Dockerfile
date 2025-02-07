# Dockerfile
# Uses multi-stage builds requiring Docker 17.05 or higher
# See https://docs.docker.com/develop/develop-images/multistage-build/

# Creating a python base with shared environment variables
# NOTICE: at this moment, 3.13 is released but gil-free version is still experimental
# However will experiment with 3.13 gil-free version to see if it is safe to use in context of AI applications
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bullseye AS python-base
ENV PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PIP_NO_CACHE_DIR=off \
PIP_DISABLE_PIP_VERSION_CHECK=on \
PIP_DEFAULT_TIMEOUT=100 \
POETRY_HOME="/opt/poetry" \
POETRY_VIRTUALENVS_IN_PROJECT=true \
POETRY_NO_INTERACTION=1 \
PYSETUP_PATH="/opt/pysetup" \
VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


ARG POETRY_VERSION=2.0.1
# builder-base is used to build dependencies
FROM python-base AS builder-base
RUN buildDeps="build-essential" \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        vim \
        netcat \
    && apt-get install -y --no-install-recommends $buildDeps \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry - respects $POETRY_VERSION & $POETRY_HOME
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=${POETRY_HOME} python3 - --version ${POETRY_VERSION} && \
    chmod a+x /opt/poetry/bin/poetry

# We copy our Python requirements here to cache them
# and install only runtime deps using poetry
WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./
RUN poetry install --only main  # respects

# 'development' stage installs all dev deps and can be used to develop code.
# For example using docker-compose to mount local volume under /app
FROM python-base as development
ENV FASTAPI_ENV=development

# Copying poetry and venv into image
COPY --from=builder-base $POETRY_HOME $POETRY_HOME
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

# # install some additional dev dependencies
# RUN buildDeps="build-essential" \
#     && apt-get update \
#     && apt-get install --no-install-recommends -y \
#         curl \
#         vim \
#         netcat \
#     && apt-get install -y --no-install-recommends $buildDeps \
#     && rm -rf /var/lib/apt/lists/*

# # Copying in our entrypoint
# COPY docker-entrypoint.sh /docker-entrypoint.sh
# RUN chmod +x /docker-entrypoint.sh

# venv already has runtime deps installed we get a quicker install
# must set workdir to pysetup path before install so that later production can pick up venv correctly
WORKDIR $PYSETUP_PATH
RUN poetry install

WORKDIR /app
COPY . .

EXPOSE 7860
# ENTRYPOINT /docker-entrypoint.sh $0 $@
# CMD ["uvicorn", "--reload", "--host=0.0.0.0", "--port=8000", "main:app"]
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860", "-h"]


# # 'lint' stage runs black and isort
# # running in check mode means build will fail if any linting errors occur
# FROM development AS lint
# RUN black --config ./pyproject.toml --check app tests
# RUN isort --settings-path ./pyproject.toml --recursive --check-only
# CMD ["tail", "-f", "/dev/null"]

# # 'test' stage runs our unit tests with pytest and
# # coverage.  Build will fail if test coverage is under 95%
# FROM development AS test
# RUN coverage run --rcfile ./pyproject.toml -m pytest ./tests
# RUN coverage report --fail-under 95

# ARG GROUPNAME=poetry

# # 'production' stage uses the clean 'python-base' stage and copyies
# # in only our runtime deps that were installed in the 'builder-base'
# FROM python-base AS production
# ENV FASTAPI_ENV=production

# COPY --from=builder-base $VENV_PATH $VENV_PATH
# COPY gunicorn_conf.py /gunicorn_conf.py

# COPY docker-entrypoint.sh /docker-entrypoint.sh
# RUN chmod +x /docker-entrypoint.sh

# # Create user with the name poetry
# RUN groupadd -g 1500 poetry && \
#     useradd -m -u 1500 -g poetry poetry

# COPY --chown=poetry:poetry ./app /app
# USER poetry
# WORKDIR /app

# ENTRYPOINT /docker-entrypoint.sh $0 $@
# CMD [ "gunicorn", "--worker-class uvicorn.workers.UvicornWorker", "--config /gunicorn_conf.py", "main:app"]