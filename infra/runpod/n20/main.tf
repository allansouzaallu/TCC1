terraform {
  required_version = ">= 1.11.4"
}

locals {
  profile = jsondecode(file("${path.module}/profile.json"))
}

output "n20_preflight" {
  description = "Provider-free capacity and scenario check; does not acquire GPU resources."
  value = {
    nodes                             = local.profile.nodes
    rounds                            = local.profile.rounds
    gpu_target                        = local.profile.gpu_target
    gpu_count                         = local.profile.gpu_count
    models_per_gpu                    = local.profile.models_per_gpu
    model                             = local.profile.model
    model_revision                    = local.profile.model_revision
    quantization                      = local.profile.quantization
    scenario_seed_pairs               = length(local.profile.scenarios) * length(local.profile.seeds)
    generations_per_scenario_and_seed = local.profile.nodes * local.profile.rounds
  }

  precondition {
    condition = (
      local.profile.nodes == 20 &&
      local.profile.rounds == 50 &&
      local.profile.gpu_count == 2 &&
      local.profile.models_per_gpu == 10 &&
      local.profile.nodes == local.profile.gpu_count * local.profile.models_per_gpu &&
      local.profile.inference_layout == "per_node" &&
      local.profile.quantization == "nf4" &&
      length(local.profile.scenarios) == 6 &&
      length(local.profile.seeds) == 5
    )
    error_message = "Expected 20 dedicated models on two GPUs, 50 rounds and 30 scenario/seed combinations."
  }
}
