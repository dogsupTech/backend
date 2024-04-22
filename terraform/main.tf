provider "google" {
	project = "vetai1994"
	region  = "europe-west1"
	zone    = "europe-west1-b"  # Example: Adjust the zone appropriately
}

provider "google-beta" {
	project = "vetai1994"
	region  = "europe-west1"
	zone    = "europe-west1-b"  # Example: Adjust the zone appropriately
}

################################################################################
# APIs to enable
################################################################################

locals {
	api_services = {
		"iam" = "iam.googleapis.com",
		"cloudresourcemanager" = "cloudresourcemanager.googleapis.com",
		"iamcredentials" = "iamcredentials.googleapis.com",
		"sts" = "sts.googleapis.com",
	}
}

resource "google_project_service" "project_services" {
	for_each = local.api_services

	provider = google-beta
	project  = "vetai1994"
	service  = each.value
}


################################################################################
# Service accounts
################################################################################

resource "google_service_account" "service_account" {
	project      = "vetai1994"
	account_id   = "github-actions-service-account"
	display_name = "Github Actions Service Account"
}



################################################################################
# GitHub Workload Identity Federation
################################################################################

module "gh_oidc" {
	source      = "terraform-google-modules/github-actions-runners/google//modules/gh-oidc"
	project_id  = "vetai1994"
	pool_id     = "gh-pool"
	provider_id = "gh-provider"
	sa_mapping  = {
		"gh-service-account" = {
			sa_name   = google_service_account.service_account.name
			attribute = "attribute.repository/dogg3/dogsupTech"
		}
	}
}


output "gh_oidc_pool_name" {
	value = module.gh_oidc.pool_name
}

output "gh_oidc_provider_name" {
	value = module.gh_oidc.provider_name
}
