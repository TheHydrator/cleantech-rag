provider "hcloud" {
  token = var.hcloud_token
}

resource "hcloud_ssh_key" "admin" {
  name       = "rag-admin"
  public_key = file(pathexpand(var.ssh_public_key_path))
}

resource "hcloud_firewall" "rag" {
  name = "rag-firewall"

  # SSH: only from your IP
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = [var.admin_ip]
  }

  # Kubernetes API: only from your IP
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "6443"
    source_ips = [var.admin_ip]
  }

  # Web traffic: open to everyone
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "80"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "443"
    source_ips = ["0.0.0.0/0", "::/0"]
  }
}

resource "hcloud_server" "rag" {
  name         = "rag-server"
  server_type  = var.server_type
  image        = "ubuntu-24.04"
  location     = var.location
  ssh_keys     = [hcloud_ssh_key.admin.id]
  firewall_ids = [hcloud_firewall.rag.id]

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }
}