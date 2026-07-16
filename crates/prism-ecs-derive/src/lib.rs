extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Derive macro implementing `prism_ecs_core::component::SchemaGoverned`.
///
/// Generates `fn field_names() -> &'static [&'static str]` returning the
/// declared field names of the struct, enabling runtime enforcement of
/// `additionalProperties: false`.
#[proc_macro_derive(SchemaGoverned)]
pub fn derive_schema_governed(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let field_names: Vec<String> = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => fields
                .named
                .iter()
                .map(|f| f.ident.as_ref().unwrap().to_string())
                .collect(),
            Fields::Unnamed(_) => {
                return syn::Error::new_spanned(&ast, "SchemaGoverned requires named fields")
                    .to_compile_error()
                    .into();
            }
            Fields::Unit => Vec::new(),
        },
        Data::Enum(_) | Data::Union(_) => {
            return syn::Error::new_spanned(&ast, "SchemaGoverned only supports structs")
                .to_compile_error()
                .into();
        }
    };

    let expanded = quote! {
        impl ::prism_ecs_core::component::SchemaGoverned for #name {
            fn field_names() -> &'static [&'static str] {
                static NAMES: &[&str] = &[#(#field_names),*];
                NAMES
            }
        }
    };

    TokenStream::from(expanded)
}
