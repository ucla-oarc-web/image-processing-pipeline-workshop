FROM public.ecr.aws/lambda/python:3.14

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY index.py prompt.txt ${LAMBDA_TASK_ROOT}/

CMD ["index.lambda_handler"]
