from model import mCodeGPT
import pandas as pd
import argparse
import os

from openai import OpenAI, AzureOpenAI

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ontoNxGPT',
        description='Standardize free-text data using ontology',
        epilog='Text at the bottom of help')

    df_ontology = pd.read_excel('./ontology/mcode_structure.xlsx', sheet_name="Ontology")
    df_prompt = pd.read_excel('./ontology/mcode_structure.xlsx', sheet_name="Prompt")
    df_promptYesNo = pd.read_excel('./ontology/mcode_structure.xlsx', sheet_name="Prompt(yesno)")

    parser.add_argument('-i','--input_file', help="Specify the input file for your program. For example, './input_file.txt'")
    parser.add_argument('-k', '--api_key', help="Specify the OpenAI API key for your program or set the OPENAI_API_KEY environment variable")
    parser.add_argument('-b', '--api_base', help="Specify a custom OpenAI API base. Required for Azure OpenAI.")
    parser.add_argument('-v', '--api_version', help="Specify the Azure OpenAI API version, for example, '2023-05-15'")
    parser.add_argument('-d', '--deployment_name', help="Specify the Azure OpenAI deployment name, for example, 'mcodegpt_gpt_35'")
    parser.add_argument('--model', help="Specify an OpenAI model name when using the standard OpenAI API, for example, 'gpt-4o-mini'")
    parser.add_argument('-m', '--method', help="Specify the prompt generating algorithm for your program, for example, 'RLS', 'BFOP', '2POP'")
    parser.add_argument('-o', '--output', help="Specify the output file name")
    
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        input_text = f.read()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("An OpenAI API key must be provided via --api_key or the OPENAI_API_KEY environment variable.")

    azure_fields = [args.api_base, args.api_version, args.deployment_name]
    use_azure = all(azure_fields)

    if any(field is not None for field in azure_fields) and not use_azure:
        parser.error("To use Azure OpenAI, you must supply --api_base, --api_version, and --deployment_name together.")

    model_name = args.model
    deployment_name = args.deployment_name

    if use_azure:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=args.api_version,
            azure_endpoint=args.api_base,
        )
        model_identifier = deployment_name
    else:
        if not model_name:
            parser.error("Standard OpenAI usage requires --model to specify the target model.")

        client_kwargs = {"api_key": api_key}
        if args.api_base:
            client_kwargs["base_url"] = args.api_base
        client = OpenAI(**client_kwargs)
        model_identifier = model_name

    model = mCodeGPT(
        df_ontology,
        df_prompt,
        df_promptYesNo,
        model_identifier,
        input_text,
        args.method,
        client,
    )

    df_result = model.run()

    df_result.to_csv('./output/' + args.output + '.csv')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
