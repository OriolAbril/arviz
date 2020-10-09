(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("e0aea153-0fae-4a7b-90c7-e83e0e6edfc3");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'e0aea153-0fae-4a7b-90c7-e83e0e6edfc3' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"b0002355-182a-4a85-bea8-92b15487b8a2":{"roots":{"references":[{"attributes":{"fill_color":{"value":"grey"},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1717","type":"Triangle"},{"attributes":{"data":{"xs":[[-32.07924633297332,-29.295334303806303],[-32.143039521681885,-29.477708831518214]],"ys":[[0.0,0.0],[-1.0,-1.0]]},"selected":{"id":"1753"},"selection_policy":{"id":"1754"}},"id":"1731","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"1716"},"glyph":{"id":"1717"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1718"},"selection_glyph":null,"view":{"id":"1720"}},"id":"1719","type":"GlyphRenderer"},{"attributes":{"text":""},"id":"1742","type":"Title"},{"attributes":{"toolbars":[{"id":"1705"}],"tools":[{"id":"1695"},{"id":"1696"},{"id":"1697"},{"id":"1698"},{"id":"1699"},{"id":"1700"},{"id":"1701"},{"id":"1702"}]},"id":"1761","type":"ProxyToolbar"},{"attributes":{"data":{"x":[-32.37106695144684,-32.71848009989285],"y":[0.0,-1.0]},"selected":{"id":"1755"},"selection_policy":{"id":"1756"}},"id":"1736","type":"ColumnDataSource"},{"attributes":{},"id":"1744","type":"BasicTickFormatter"},{"attributes":{"data":{"xs":[[-30.896420573800537,-30.724327779399562]],"ys":[[-0.75,-0.75]]},"selected":{"id":"1749"},"selection_policy":{"id":"1750"}},"id":"1721","type":"ColumnDataSource"},{"attributes":{"callback":null},"id":"1702","type":"HoverTool"},{"attributes":{"dimension":"height","line_color":"grey","line_dash":[6],"line_width":1.875,"location":-30.687290318389813},"id":"1741","type":"Span"},{"attributes":{},"id":"1698","type":"WheelZoomTool"},{"attributes":{},"id":"1751","type":"Selection"},{"attributes":{"children":[[{"id":"1678"},0,0]]},"id":"1760","type":"GridBox"},{"attributes":{"children":[{"id":"1762"},{"id":"1760"}]},"id":"1763","type":"Column"},{"attributes":{"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1722","type":"MultiLine"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"grey"},"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1718","type":"Triangle"},{"attributes":{"data_source":{"id":"1726"},"glyph":{"id":"1727"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1728"},"selection_glyph":null,"view":{"id":"1730"}},"id":"1729","type":"GlyphRenderer"},{"attributes":{"ticks":[0.0,-0.75,-1.0]},"id":"1714","type":"FixedTicker"},{"attributes":{},"id":"1752","type":"UnionRenderers"},{"attributes":{"source":{"id":"1716"}},"id":"1720","type":"CDSView"},{"attributes":{"fill_color":{"value":null},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1727","type":"Circle"},{"attributes":{},"id":"1754","type":"UnionRenderers"},{"attributes":{"source":{"id":"1726"}},"id":"1730","type":"CDSView"},{"attributes":{},"id":"1688","type":"BasicTicker"},{"attributes":{},"id":"1700","type":"UndoTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1695"},{"id":"1696"},{"id":"1697"},{"id":"1698"},{"id":"1699"},{"id":"1700"},{"id":"1701"},{"id":"1702"}]},"id":"1705","type":"Toolbar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"1704","type":"PolyAnnotation"},{"attributes":{"source":{"id":"1731"}},"id":"1735","type":"CDSView"},{"attributes":{"overlay":{"id":"1704"}},"id":"1699","type":"LassoSelectTool"},{"attributes":{},"id":"1753","type":"Selection"},{"attributes":{},"id":"1685","type":"LinearScale"},{"attributes":{"formatter":{"id":"1746"},"major_label_overrides":{"-0.75":"","-1":"Centered 8 schools","0":"Non-centered 8 schools"},"ticker":{"id":"1714"}},"id":"1691","type":"LinearAxis"},{"attributes":{"axis":{"id":"1687"},"ticker":null},"id":"1690","type":"Grid"},{"attributes":{},"id":"1683","type":"LinearScale"},{"attributes":{"data":{"x":[-30.81037417660005],"y":[-0.75]},"selected":{"id":"1747"},"selection_policy":{"id":"1748"}},"id":"1716","type":"ColumnDataSource"},{"attributes":{},"id":"1746","type":"BasicTickFormatter"},{"attributes":{},"id":"1695","type":"ResetTool"},{"attributes":{},"id":"1701","type":"SaveTool"},{"attributes":{"source":{"id":"1721"}},"id":"1725","type":"CDSView"},{"attributes":{},"id":"1749","type":"Selection"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1703","type":"BoxAnnotation"},{"attributes":{},"id":"1755","type":"Selection"},{"attributes":{"toolbar":{"id":"1761"},"toolbar_location":"above"},"id":"1762","type":"ToolbarBox"},{"attributes":{"data_source":{"id":"1731"},"glyph":{"id":"1732"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1733"},"selection_glyph":null,"view":{"id":"1735"}},"id":"1734","type":"GlyphRenderer"},{"attributes":{},"id":"1748","type":"UnionRenderers"},{"attributes":{},"id":"1756","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"black"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1737","type":"Circle"},{"attributes":{"axis":{"id":"1691"},"dimension":1,"ticker":null},"id":"1694","type":"Grid"},{"attributes":{"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1732","type":"MultiLine"},{"attributes":{},"id":"1679","type":"DataRange1d"},{"attributes":{"line_alpha":{"value":0.1},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1733","type":"MultiLine"},{"attributes":{"axis_label":"Log","formatter":{"id":"1744"},"ticker":{"id":"1688"}},"id":"1687","type":"LinearAxis"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":null},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1728","type":"Circle"},{"attributes":{"below":[{"id":"1687"}],"center":[{"id":"1690"},{"id":"1694"}],"left":[{"id":"1691"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"1719"},{"id":"1724"},{"id":"1729"},{"id":"1734"},{"id":"1739"},{"id":"1741"}],"title":{"id":"1742"},"toolbar":{"id":"1705"},"toolbar_location":null,"x_range":{"id":"1679"},"x_scale":{"id":"1683"},"y_range":{"id":"1681"},"y_scale":{"id":"1685"}},"id":"1678","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1747","type":"Selection"},{"attributes":{"data_source":{"id":"1721"},"glyph":{"id":"1722"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1723"},"selection_glyph":null,"view":{"id":"1725"}},"id":"1724","type":"GlyphRenderer"},{"attributes":{"data":{"x":[-30.687290318389813,-30.81037417660005],"y":[0.0,-1.0]},"selected":{"id":"1751"},"selection_policy":{"id":"1752"}},"id":"1726","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"black"},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1738","type":"Circle"},{"attributes":{},"id":"1750","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"1736"},"glyph":{"id":"1737"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1738"},"selection_glyph":null,"view":{"id":"1740"}},"id":"1739","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"1703"}},"id":"1697","type":"BoxZoomTool"},{"attributes":{},"id":"1696","type":"PanTool"},{"attributes":{"source":{"id":"1736"}},"id":"1740","type":"CDSView"},{"attributes":{"end":0.5,"start":-1.5},"id":"1681","type":"DataRange1d"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1723","type":"MultiLine"}],"root_ids":["1763"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"b0002355-182a-4a85-bea8-92b15487b8a2","root_ids":["1763"],"roots":{"1763":"e0aea153-0fae-4a7b-90c7-e83e0e6edfc3"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();