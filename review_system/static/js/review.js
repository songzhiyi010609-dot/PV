(function () {
  const picker = document.getElementById("satellite-picker");
  const image = document.getElementById("satellite-image");
  const lonInput = document.getElementById("final_lon");
  const latInput = document.getElementById("final_lat");
  const manualMarker = document.getElementById("manual-marker");
  const polygonInput = document.getElementById("final_polygon_geojson");
  const polygonModeButton = document.getElementById("polygon-mode");
  const pointModeButton = document.getElementById("point-mode");
  const clearPolygonButton = document.getElementById("clear-polygon");
  const showAllMarkersButton = document.getElementById("show-all-markers");
  const polygonElement = document.getElementById("review-polygon");
  const polylineElement = document.getElementById("review-polyline");
  const verticesElement = document.getElementById("polygon-vertices");
  const detailPanel = document.getElementById("poi-detail");
  const detailEmpty = document.getElementById("poi-detail-empty");
  const useDetailPointButton = document.getElementById("use-detail-point");
  const detailFocusMarker = document.getElementById("detail-focus-marker");
  const onlineMapElement = document.getElementById("online-map");
  const loadOnlineMapButton = document.getElementById("load-online-map");
  const syncOnlineMapButton = document.getElementById("sync-online-map");

  if (!picker || !image || !lonInput || !latInput) {
    return;
  }

  const centerLon = Number(picker.dataset.centerLon);
  const centerLat = Number(picker.dataset.centerLat);
  const zoom = Number(picker.dataset.zoom);
  const size = Number(picker.dataset.size);
  const tileSize = 256;
  const scale = tileSize * Math.pow(2, zoom);
  let mode = "point";
  let polygonPoints = [];
  let selectedDetailPoint = null;
  let onlineMap = null;
  let onlineMapMarker = null;
  let onlineMapLoading = false;
  const svgNamespace = "http://www.w3.org/2000/svg";

  function lonLatToPixel(lon, lat) {
    const clippedLat = Math.max(Math.min(lat, 85.05112878), -85.05112878);
    const x = ((lon + 180) / 360) * scale;
    const latRad = (clippedLat * Math.PI) / 180;
    const y = (0.5 - Math.log((1 + Math.sin(latRad)) / (1 - Math.sin(latRad))) / (4 * Math.PI)) * scale;
    return { x, y };
  }

  function pixelToLonLat(x, y) {
    const center = lonLatToPixel(centerLon, centerLat);
    const globalX = center.x + (x - size / 2);
    const globalY = center.y + (y - size / 2);
    const lon = (globalX / scale) * 360 - 180;
    const mercY = 0.5 - globalY / scale;
    const lat = 90 - (360 * Math.atan(Math.exp(-mercY * 2 * Math.PI))) / Math.PI;
    return { lon, lat };
  }

  function lonLatToImagePosition(lon, lat) {
    const center = lonLatToPixel(centerLon, centerLat);
    const point = lonLatToPixel(Number(lon), Number(lat));
    const x = size / 2 + (point.x - center.x);
    const y = size / 2 + (point.y - center.y);
    return {
      x,
      y,
      leftPct: (x / size) * 100,
      topPct: (y / size) * 100,
      inBounds: x >= 0 && x <= size && y >= 0 && y <= size,
    };
  }

  function clearDetailFocus() {
    picker.classList.remove("detail-focus-mode");
    if (detailFocusMarker) {
      detailFocusMarker.hidden = true;
      detailFocusMarker.textContent = "";
      detailFocusMarker.className = "marker detail-focus";
    }
  }

  function setManualPoint(lon, lat, options = {}) {
    if (!options.keepFocus) {
      clearDetailFocus();
    }
    lonInput.value = Number(lon).toFixed(8);
    latInput.value = Number(lat).toFixed(8);
    const position = lonLatToImagePosition(lon, lat);
    manualMarker.style.left = `${position.leftPct}%`;
    manualMarker.style.top = `${position.topPct}%`;
    manualMarker.hidden = false;
  }

  function providerName(value) {
    const names = { amap: "高德", tencent: "腾讯", baidu: "百度" };
    return names[value] || value || "";
  }

  function focusDetailPoint(source) {
    picker.classList.add("detail-focus-mode");
    if (!detailFocusMarker) {
      return { inBounds: false };
    }
    const lon = Number(source.dataset.lon);
    const lat = Number(source.dataset.lat);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      clearDetailFocus();
      return { inBounds: false };
    }
    const position = lonLatToImagePosition(lon, lat);
    const provider = source.dataset.provider || "";
    const clampedLeft = Math.max(2, Math.min(98, position.leftPct));
    const clampedTop = Math.max(2, Math.min(98, position.topPct));
    detailFocusMarker.className = `marker detail-focus provider-${provider}`;
    detailFocusMarker.classList.toggle("offscreen", !position.inBounds);
    detailFocusMarker.style.left = `${position.inBounds ? position.leftPct : clampedLeft}%`;
    detailFocusMarker.style.top = `${position.inBounds ? position.topPct : clampedTop}%`;
    detailFocusMarker.textContent = provider.slice(0, 1).toUpperCase() || "P";
    detailFocusMarker.title = `${providerName(provider)} #${source.dataset.rank || ""} ${source.dataset.poiName || ""}`;
    detailFocusMarker.hidden = false;
    return { inBounds: position.inBounds };
  }

  function setDetailField(name, value) {
    const element = detailPanel?.querySelector(`[data-field="${name}"]`);
    if (element) {
      element.textContent = value || "-";
    }
  }

  function riskExplanation(flags, poiType, poiName) {
    const text = `${flags || ""} ${poiType || ""} ${poiName || ""}`;
    const risks = [];
    if (text.includes("internal_sub_poi")) risks.push("可能是商场内部店铺/服务点");
    if (text.includes("bad_place_hint")) risks.push("可能是停车场、入口、交通点或非主体");
    if (text.includes("停车场")) risks.push("停车场点，不宜当作商场中心");
    if (text.includes("入口") || text.includes("出口") || text.includes("门")) risks.push("门/出入口点，通常偏离主体中心");
    if (text.includes("地铁")) risks.push("靠近地铁相关描述，需确认不是站点");
    if (text.includes("办公")) risks.push("办公区点，需确认是否代表商场主体");
    return risks.join("；");
  }

  function showPoiDetail(source) {
    selectedDetailPoint = {
      lon: source.dataset.lon,
      lat: source.dataset.lat,
    };
    const provider = source.dataset.provider || "";
    const rank = source.dataset.rank || "";
    const poiName = source.dataset.poiName || "";
    const address = source.dataset.address || "";
    const poiType = source.dataset.poiType || "";
    const score = source.dataset.score || "";
    const flags = source.dataset.flags || "";
    const focusResult = focusDetailPoint(source);
    const risk = riskExplanation(flags, poiType, poiName);
    const visibilityNote = focusResult.inBounds ? "" : "该候选点不在当前卫星图范围内，已隐藏其他点；请结合坐标和外部地图复核。";
    detailEmpty.hidden = true;
    detailPanel.hidden = false;
    useDetailPointButton.hidden = false;
    setDetailField("provider", providerName(provider));
    setDetailField("rank", rank);
    setDetailField("poi_name", poiName);
    setDetailField("address", address);
    setDetailField("poi_type", poiType);
    setDetailField("coord", `${Number(source.dataset.lon).toFixed(8)}, ${Number(source.dataset.lat).toFixed(8)}`);
    setDetailField("score", score);
    setDetailField("flags", [visibilityNote, risk || flags || "无明显风险标记"].filter(Boolean).join("；"));
  }

  function setMode(nextMode) {
    mode = nextMode;
    pointModeButton?.classList.toggle("active", mode === "point");
    polygonModeButton?.classList.toggle("active", mode === "polygon");
  }

  function polygonCentroid(points) {
    if (points.length === 0) {
      return null;
    }
    if (points.length < 3) {
      const avg = points.reduce((acc, item) => ({ x: acc.x + item.x, y: acc.y + item.y }), { x: 0, y: 0 });
      return { x: avg.x / points.length, y: avg.y / points.length };
    }
    let area = 0;
    let cx = 0;
    let cy = 0;
    for (let i = 0; i < points.length; i += 1) {
      const current = points[i];
      const next = points[(i + 1) % points.length];
      const cross = current.x * next.y - next.x * current.y;
      area += cross;
      cx += (current.x + next.x) * cross;
      cy += (current.y + next.y) * cross;
    }
    area /= 2;
    if (Math.abs(area) < 1e-6) {
      const avg = points.reduce((acc, item) => ({ x: acc.x + item.x, y: acc.y + item.y }), { x: 0, y: 0 });
      return { x: avg.x / points.length, y: avg.y / points.length };
    }
    return { x: cx / (6 * area), y: cy / (6 * area) };
  }

  function updatePolygon() {
    const pointString = polygonPoints.map((point) => `${(point.x / size) * 100},${(point.y / size) * 100}`).join(" ");
    polygonElement?.setAttribute("points", polygonPoints.length >= 3 ? pointString : "");
    polylineElement?.setAttribute("points", pointString);
    if (verticesElement) {
      verticesElement.replaceChildren();
      polygonPoints.forEach((point, index) => {
        const cx = (point.x / size) * 100;
        const cy = (point.y / size) * 100;
        const circle = document.createElementNS(svgNamespace, "circle");
        circle.setAttribute("cx", String(cx));
        circle.setAttribute("cy", String(cy));
        circle.setAttribute("r", "1.25");
        circle.setAttribute("class", "polygon-vertex");
        verticesElement.appendChild(circle);

        const label = document.createElementNS(svgNamespace, "text");
        label.setAttribute("x", String(cx + 1.6));
        label.setAttribute("y", String(cy - 1.6));
        label.setAttribute("class", "polygon-vertex-label");
        label.textContent = String(index + 1);
        verticesElement.appendChild(label);
      });
    }
    if (polygonPoints.length >= 3) {
      const centroid = polygonCentroid(polygonPoints);
      const coordinates = polygonPoints.map((point) => [Number(point.lon.toFixed(8)), Number(point.lat.toFixed(8))]);
      coordinates.push(coordinates[0]);
      polygonInput.value = JSON.stringify({
        type: "Polygon",
        coordinates: [coordinates],
      });
      if (centroid) {
        const center = pixelToLonLat(centroid.x, centroid.y);
        setManualPoint(center.lon, center.lat);
      }
    } else {
      polygonInput.value = "";
    }
  }

  function addPolygonPoint(x, y) {
    const point = pixelToLonLat(x, y);
    polygonPoints.push({ x, y, lon: point.lon, lat: point.lat });
    updatePolygon();
  }

  function clearPolygon() {
    clearDetailFocus();
    polygonPoints = [];
    polygonInput.value = "";
    polygonElement?.setAttribute("points", "");
    polylineElement?.setAttribute("points", "");
    verticesElement?.replaceChildren();
  }

  pointModeButton?.addEventListener("click", () => setMode("point"));
  polygonModeButton?.addEventListener("click", () => setMode("polygon"));
  clearPolygonButton?.addEventListener("click", clearPolygon);
  showAllMarkersButton?.addEventListener("click", clearDetailFocus);

  image.addEventListener("click", function (event) {
    const rect = image.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * size;
    const y = ((event.clientY - rect.top) / rect.height) * size;
    if (mode === "polygon") {
      clearDetailFocus();
      addPolygonPoint(x, y);
    } else {
      const point = pixelToLonLat(x, y);
      setManualPoint(point.lon, point.lat);
    }
  });

  document.querySelectorAll(".marker.provider,.show-candidate").forEach(function (button) {
    button.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      showPoiDetail(button);
    });
  });

  document.querySelectorAll(".pick-candidate").forEach(function (button) {
    button.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      showPoiDetail(button);
      setManualPoint(button.dataset.lon, button.dataset.lat, { keepFocus: true });
    });
  });

  useDetailPointButton?.addEventListener("click", function () {
    if (selectedDetailPoint) {
      setManualPoint(selectedDetailPoint.lon, selectedDetailPoint.lat, { keepFocus: true });
    }
  });

  function restoreExistingPolygon() {
    const raw = picker.dataset.existingPolygon;
    if (!raw) {
      return;
    }
    try {
      const parsed = JSON.parse(raw);
      const coords = parsed.coordinates?.[0] || [];
      const center = lonLatToPixel(centerLon, centerLat);
      polygonPoints = coords.slice(0, -1).map(([lon, lat]) => {
        const point = lonLatToPixel(Number(lon), Number(lat));
        return {
          lon: Number(lon),
          lat: Number(lat),
          x: size / 2 + (point.x - center.x),
          y: size / 2 + (point.y - center.y),
        };
      });
      updatePolygon();
    } catch (error) {
      polygonPoints = [];
    }
  }

  function currentFormPoint() {
    const lon = Number(lonInput.value || onlineMapElement?.dataset.centerLon);
    const lat = Number(latInput.value || onlineMapElement?.dataset.centerLat);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      return null;
    }
    return { lon, lat };
  }

  function setOnlineMapPoint(lon, lat, shouldWriteInputs = true) {
    if (!onlineMap || !window.AMap) {
      return;
    }
    const point = [Number(lon), Number(lat)];
    if (!onlineMapMarker) {
      onlineMapMarker = new window.AMap.Marker({
        position: point,
        draggable: true,
        title: "最终中心点",
      });
      onlineMap.add(onlineMapMarker);
      onlineMapMarker.on("dragend", function (event) {
        const position = event.lnglat;
        setManualPoint(position.lng, position.lat);
      });
    } else {
      onlineMapMarker.setPosition(point);
    }
    onlineMap.setCenter(point);
    if (shouldWriteInputs) {
      setManualPoint(point[0], point[1]);
    }
  }

  function initializeOnlineMap(config) {
    if (!onlineMapElement || !window.AMap || onlineMap) {
      return;
    }
    const initialPoint = currentFormPoint() || { lon: 121.4737, lat: 31.2304 };
    onlineMapElement.classList.add("online-map-loaded");
    onlineMap = new window.AMap.Map("online-map", {
      zoom: Number(config.defaultZoom || 18),
      center: [initialPoint.lon, initialPoint.lat],
      viewMode: "2D",
    });
    setOnlineMapPoint(initialPoint.lon, initialPoint.lat, Boolean(lonInput.value && latInput.value));
    onlineMap.on("click", function (event) {
      setOnlineMapPoint(event.lnglat.lng, event.lnglat.lat);
    });
  }

  function loadScript(src, onload, onerror) {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = onload;
    script.onerror = onerror;
    document.head.appendChild(script);
  }

  async function loadOnlineMap() {
    if (!onlineMapElement || onlineMapLoading) {
      return;
    }
    onlineMapLoading = true;
    loadOnlineMapButton && (loadOnlineMapButton.disabled = true);
    try {
      const response = await fetch("/api/map_config");
      const config = await response.json();
      if (!config.enabled) {
        onlineMapElement.innerHTML = '<div class="empty-image">在线地图未启用，请使用卫星图或候选点选点。</div>';
        return;
      }
      if (!config.amapKey) {
        onlineMapElement.innerHTML = '<div class="empty-image">缺少高德 Web JS key。请在 review_config.json 的 online_map.amap_key 中配置；也可先用外部地图复制坐标。</div>';
        return;
      }
      if (window.AMap) {
        initializeOnlineMap(config);
        return;
      }
      loadScript(
        `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(config.amapKey)}`,
        () => initializeOnlineMap(config),
        () => {
          onlineMapElement.innerHTML = '<div class="empty-image">在线地图加载失败。请检查网络或高德 key。</div>';
        }
      );
    } catch (error) {
      onlineMapElement.innerHTML = '<div class="empty-image">在线地图配置读取失败，请刷新后重试。</div>';
    } finally {
      onlineMapLoading = false;
      loadOnlineMapButton && (loadOnlineMapButton.disabled = false);
    }
  }

  loadOnlineMapButton?.addEventListener("click", loadOnlineMap);
  syncOnlineMapButton?.addEventListener("click", function () {
    if (!onlineMap) {
      loadOnlineMap();
      return;
    }
    const point = currentFormPoint();
    if (point) {
      setOnlineMapPoint(point.lon, point.lat, false);
    }
  });

  restoreExistingPolygon();
})();
