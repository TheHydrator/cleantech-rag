variable "hcloud_token" {
  description = "Hetzner Cloud API token"
  type        = string
  sensitive   = true
}

variable "admin_ip" {
  description = "Your public IP in CIDR form, e.g. 203.0.113.7/32. Only this IP can SSH in or reach Kubernetes."
  type        = string
}

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "server_type" {
  type    = string
  default = "cx33"
}

variable "location" {
  type    = string
  default = "fsn1" # Falkenstein, Germany
}